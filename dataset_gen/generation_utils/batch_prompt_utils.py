import gc
import torch
import logging
import os
import json
import time
from math import ceil
from typing import List, Dict
from PIL import Image
import bittensor as bt
from transformers import logging as transformers_logging

def batch_process_dataset(dataset, start_index, dataset_name, prompt_generator, annotations_dir, batch_size=16, annotation_task='t2i'):
    """
    Process a dataset in batches, generating and saving annotations.
    Uses a two-phase approach to minimize model loading/unloading:
    1. Process all images with VLM
    2. Process all descriptions with LLM
    
    Args:
        dataset: The dataset to process
        start_index: Starting index for the dataset
        dataset_name: Name of the dataset
        prompt_generator: PromptGenerator instance to use
        annotations_dir: Directory to save annotations
        batch_size: Size of batches to process
        annotation_task: Task type for processing descriptions
        
    Returns:
        Number of images processed
    """
    image_count = 0
    start_time = time.time()
    total_images = len(dataset)
    
    # Check for existing annotations to resume processing
    existing_annotations = set()
    if os.path.exists(annotations_dir):
        for filename in os.listdir(annotations_dir):
            if filename.endswith('.json'):
                try:
                    idx = int(filename.split('.')[0])
                    existing_annotations.add(idx)
                except ValueError:
                    pass
    
    print(f"Found {len(existing_annotations)} existing annotations, will skip these.")
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = min(1000, total_images)  # Process up to 1000 images at a time
    
    for chunk_start in range(0, total_images, chunk_size):
        chunk_timer_start = time.time()
        chunk_end = min(chunk_start + chunk_size, total_images)
        print(f"Processing chunk {chunk_start//chunk_size + 1}/{ceil(total_images/chunk_size)} ({chunk_start}-{chunk_end-1})")
        
        # Get chunk of images and indices
        chunk_images = []
        chunk_indices = []
        for i in range(chunk_start, chunk_end):
            real_image = dataset[i]
            adjusted_index = i + start_index
            
            # Skip if annotation already exists
            if adjusted_index in existing_annotations:
                continue
                
            chunk_images.append(real_image['image'])
            chunk_indices.append(adjusted_index)
        
        if not chunk_images:
            print(f"Skipping chunk {chunk_start//chunk_size + 1} - all annotations already exist")
            continue
            
        print(f"Processing {len(chunk_images)} images in this chunk")
        
        # PHASE 1: Process this chunk with VLM
        print(f"PHASE 1: Processing images with VLM...")
        
        # Load VLM once for this chunk
        original_device = prompt_generator.device
        print(f"Loading VLM model {prompt_generator.vlm_name} on {original_device}")
        prompt_generator.load_vlm()
        
        # Process all images in batches
        raw_descriptions = []
        with torch.no_grad():  # Reduce memory usage during inference
            for i in range(0, len(chunk_images), batch_size):
                batch_end = min(i + batch_size, len(chunk_images))
                batch_images = chunk_images[i:batch_end]
                
                print(f"Processing VLM batch {i//batch_size + 1}/{ceil(len(chunk_images)/batch_size)}")
                
                batch_descriptions = []
                for j, image in enumerate(batch_images):
                    description = ""
                    prompts = [
                        "An image of",
                        "The setting is",
                        "The background is",
                        "The image type/style is"
                    ]

                    for prompt_idx, prompt in enumerate(prompts):
                        description += prompt + ' '
                        inputs = prompt_generator.vlm_processor(
                            image,
                            text=description,
                            return_tensors="pt"
                        ).to(original_device, torch.float16)

                        generated_ids = prompt_generator.vlm.generate(
                            **inputs,
                            max_new_tokens=20
                        )
                        answer = prompt_generator.vlm_processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0].strip()

                        if answer:
                            answer = answer.rstrip(" ,;!?")
                            if not answer.endswith('.'):
                                answer += '.'
                            description += answer + ' '
                        else:
                            description = description[:-len(prompt) - 1]

                    if description.startswith(prompts[0]):
                        description = description[len(prompts[0]):]

                    description = description.strip()
                    if not description.endswith('.'):
                        description += '.'
                        
                    batch_descriptions.append(description)
                
                raw_descriptions.extend(batch_descriptions)
                print(f"Processed {batch_end}/{len(chunk_images)} images with VLM")
        
        # Clear VLM from memory
        print("Unloading VLM from GPU memory")
        prompt_generator.clear_gpu()
        
        # PHASE 2: Process all descriptions with LLM
        print(f"PHASE 2: Processing descriptions with LLM...")
        
        # Load LLM once for this chunk
        print(f"Loading LLM model {prompt_generator.llm_name} on {original_device}")
        try:
            prompt_generator.load_llm()
            
            # Process all descriptions in batches
            with torch.no_grad():  # Reduce memory usage during inference
                for i in range(0, len(raw_descriptions), batch_size):
                    batch_end = min(i + batch_size, len(raw_descriptions))
                    batch_descriptions = raw_descriptions[i:batch_end]
                    batch_indices = chunk_indices[i:batch_end]
                    
                    print(f"Processing LLM batch {i//batch_size + 1}/{ceil(len(raw_descriptions)/batch_size)}")
                    
                    batch_final = []
                    for description in batch_descriptions:
                        try:
                            moderated_description = prompt_generator.moderate(description)
                            if annotation_task in ['t2v', 'i2v']:
                                final_desc = prompt_generator.enhance(moderated_description)
                            else:
                                final_desc = moderated_description
                            batch_final.append(final_desc)
                        except Exception as e:
                            print(f"Error processing description: {e}")
                            batch_final.append(description)
                    
                    # Save this batch of annotations immediately
                    for idx, final_desc in zip(batch_indices, batch_final):
                        annotation = {
                            "id": idx,
                            "dataset": dataset_name,
                            "description": final_desc
                        }
                        file_path = os.path.join(annotations_dir, f"{idx}.json")
                        with open(file_path, 'w') as f:
                            json.dump(annotation, f)
                    
                    image_count += len(batch_indices)
                    print(f"Progress: {image_count}/{total_images} annotations generated and saved.")
        except Exception as e:
            print(f"Error during LLM processing: {e}")
            # Save raw descriptions as fallback if LLM processing fails
            print("Saving raw descriptions as fallback...")
            for idx, raw_desc in zip(chunk_indices, raw_descriptions):
                annotation = {
                    "id": idx,
                    "dataset": dataset_name,
                    "description": raw_desc
                }
                file_path = os.path.join(annotations_dir, f"{idx}.json")
                with open(file_path, 'w') as f:
                    json.dump(annotation, f)
            
            image_count += len(chunk_indices)
        finally:
            # Clear LLM from memory
            print("Unloading LLM from GPU memory")
            prompt_generator.clear_gpu()
        
        # Clear chunk data to free memory
        chunk_images = None
        raw_descriptions = None
        gc.collect()
        chunk_timer_end = time.time()
        print(f"Chunk {chunk_start//chunk_size + 1} processed in {chunk_timer_end - chunk_timer_start:.2f} seconds.")
    
    duration = time.time() - start_time
    print(f"All {image_count} annotations generated and saved in {duration:.2f} seconds.")
    if image_count > 0:
        print(f"Mean annotation generation time: {duration/image_count:.2f} seconds.")
    
    return image_count