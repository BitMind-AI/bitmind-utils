import argparse
import json
import os
import torch
import time
from pathlib import Path
import pandas as pd
from math import ceil
from PIL import Image
import copy
import requests
import imghdr
import io
import numpy as np
import cv2
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import load_dataset
from dataset_utils.hugging_face_utils import (
    dataset_exists_on_hf,
    load_and_sort_dataset,
    slice_dataset,
    save_as_json,
    upload_videos_as_files
)
from generation_utils.batch_prompt_utils import batch_process_dataset
from diffusers.utils import export_to_video
from bitmind.generation.prompt_generator import PromptGenerator
from bitmind.generation.generation_pipeline import GenerationPipeline, IMAGE_ANNOTATION_MODEL, TEXT_MODERATION_MODEL
from bitmind.generation.models import initialize_model_registry

PROGRESS_INCREMENT = 10


def parse_arguments():
    """Parse command-line arguments for generating synthetic images and annotations.

    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login

    Do not add token as Git credential.

    Arguments:
        --hf_org (str): Required. Hugging Face organization name.
        --real_image_dataset_name (str): Required. Name of the real image dataset.
        --diffusion_model (str): Required. Diffusion model to use for image generation.
        --download_annotations (bool): Optional. Flag to download existing annotations.
        --skip_generate_annotations (bool): Optional. Flag to skip local annotation generation.
                                        Useful when local annotations exist.
        --generate_synthetic_images (bool): Optional. Flag to generate synthetic images.
        --hf_token (str): Required for interfacing with Hugging Face.
        --start_index (int): Required. Start index (inclusive) for processing the dataset.
        --end_index (int): Optional. End index (inclusive) for processing the dataset.
        --gpu_id (int): Required. Which GPU to use (check nvidia-smi -L).
        --download_real_images (bool): Optional. Download real images for i2i generation.
        --target_org (str): Optional. Target Hugging Face org for uploading annotations (default: same as --hf_org)
        --private (bool): Optional. Upload the dataset as private
        --max_images (int): Optional. Maximum number of images to annotate per dataset.
        --annotation_split (str): Optional. Which split to load from the annotation dataset (default: train)
        --local_image_dir (str): Optional. Path to a local directory of images to annotate.
    """
    parser = argparse.ArgumentParser(
        description='Generate synthetic images and annotations from a real dataset.'
    )
    parser.add_argument(
        '--hf_org',
        type=str,
        required=True,
        help='Hugging Face org name.'
    )
    parser.add_argument(
        '--real_image_dataset_name',
        type=str,
        required=True,
        help='Real image dataset name.'
    )
    parser.add_argument(
        '--diffusion_model',
        type=str,
        required=False,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help='Diffusion model to use for image generation. Only needed if --generate_synthetic_images is specified.'
    )
    parser.add_argument(
        '--download_annotations',
        action='store_true',
        default=False,
        help='Download annotations from Hugging Face.'
    )
    parser.add_argument(
        '--skip_generate_annotations',
        action='store_true',
        default=False,
        help='Skip annotation generation and use existing annotations.'
    )
    parser.add_argument(
        '--generate_synthetic_images',
        action='store_true',
        default=False,
        help='Generate synthetic images.'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='Token for uploading to Hugging Face.'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        required=True,
        help='Start index for processing the dataset. Default to the first index.'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=None,
        help='End index for processing the dataset. Default to the last index.'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        required=True,
        help='Which GPU to use (check nvidia-smi -L).'
    )
    parser.add_argument(
        '--download_real_images',
        action='store_true',
        default=False,
        help='Download real images for i2i generation.'
    )
    parser.add_argument(
        '--output_repo_name',
        type=str,
        help='Custom name for the output repository on HuggingFace. If not provided, will use auto-generated name.'
    )
    parser.add_argument(
        '--target_org',
        type=str,
        required=False,
        help='Target Hugging Face org for uploading annotations (default: same as --hf_org)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        default=False,
        help='Upload the dataset as private'
    )
    parser.add_argument(
        '--annotation_task',
        type=str,
        default='t2i',
        choices=['t2i', 'i2i', 't2v', 'i2v'],
        help='Which annotation prompt type to generate (t2i/i2i: non-enhanced, t2v/i2v: enhanced)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of images to annotate per dataset.'
    )
    parser.add_argument(
        '--annotation_split',
        type=str,
        default='train',
        help='Which split to load from the annotation dataset (default: train)'
    )
    parser.add_argument(
        '--test_mask_randomization',
        action='store_true',
        default=False,
        help='Test a variety of mask randomization settings and log mask parameters/results to a JSONL file.'
    )
    parser.add_argument(
        '--local_image_dir',
        type=str,
        default=None,
        help='Path to a local directory of images to annotate.'
    )
    return parser.parse_args()


def download_real_images(dataset, start_index, end_index, output_dir):
    """Download real images from the dataset
    Args:
        dataset: The source dataset containing images
        start_index: Starting index for processing
        end_index: Ending index for processing
        output_dir: Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    total_downloaded = 0

    # Slice the dataset to the desired range
    dataset_slice = dataset.select(range(start_index, end_index + 1))

    for idx, item in enumerate(dataset_slice):
        if 'image' not in item:
            print(f"No image found in item {start_index + idx}")
            continue

        try:
            image = item['image']
            if isinstance(image, str):  # If image is a path/url
                try:
                    image = Image.open(requests.get(image, stream=True).raw)
                except Exception as e:
                    print(f"Failed to load image at index {start_index + idx}: {e}")
                    continue
            elif not isinstance(image, Image.Image):
                try:
                    image = Image.fromarray(image)
                except Exception as e:
                    print(f"Failed to convert image at index {start_index + idx}: {e}")
                    continue

            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Determine original format if possible
            original_format = None
            if hasattr(image, 'format') and image.format:
                original_format = image.format
            else:
                # Try to guess from bytes
                try:
                    buf = io.BytesIO()
                    image.save(buf, format='PNG')
                    buf.seek(0)
                    guessed = imghdr.what(buf)
                    if guessed:
                        original_format = guessed.upper()
                    else:
                        original_format = 'PNG'
                except Exception:
                    original_format = 'PNG'

            # Use original extension if possible, else default to .png
            ext = original_format.lower() if original_format else 'png'
            image_path = os.path.join(output_dir, f"{start_index + idx}.{ext}")
            image.save(image_path, format=original_format if original_format else 'PNG')
            total_downloaded += 1

            if total_downloaded % 100 == 0:
                print(f"Downloaded and processed {total_downloaded} images")

        except Exception as e:
            print(f"Error processing image at index {start_index + idx}: {e}")
            continue

    print(f"Successfully downloaded and processed {total_downloaded} images")
    return total_downloaded


def generate_and_save_annotations(
    dataset,
    start_index,
    dataset_name,
    prompt_generator,
    annotations_dir,
    args,
    batch_size=16
):
    annotations_batch = []
    image_count = 0
    start_time = time.time()
    # Update progress every PROGRESS_INCREMENT % of image chunk
    progress_interval = (
        batch_size * ceil(len(dataset) / (PROGRESS_INCREMENT * batch_size))
    )

    # Load models ONCE before the loop
    print("Loading models for annotation generation...")
    prompt_generator.load_models()

    try:
        for index, real_image in enumerate(dataset):
            # For local datasets, the dataset is already sliced, so use the actual index
            # For HF datasets, we need to add start_index
            if args.local_image_dir:
                adjusted_index = start_index + index
            else:
                adjusted_index = index + start_index

            # Generate annotation without reloading models
            annotation = {
                "id": adjusted_index,
                "dataset": dataset_name,
                "description": prompt_generator.generate(real_image['image'], downstream_task=args.annotation_task)
            }

            annotations_batch.append((adjusted_index, annotation))

            if len(annotations_batch) == batch_size or image_count == len(dataset) - 1:
                for image_id, annotation in annotations_batch:
                    file_path = os.path.join(annotations_dir, f"{image_id}.json")
                    with open(file_path, 'w') as f:
                        json.dump(annotation, f)
                annotations_batch = []

            image_count += 1

            if image_count % progress_interval == 0 or image_count == len(dataset):
                print(f"Progress: {image_count}/{len(dataset)} annotations generated.")
    finally:
        # Ensure models are cleared even if an error occurs
        prompt_generator.clear_gpu()

    duration = time.time() - start_time
    print(
        f"All {image_count} annotations generated and saved in {duration:.2f} seconds."
    )
    print(
        f"Mean annotation generation time: {duration/image_count:.2f} seconds if any."
    )


def build_image_samples(annotations_chunk_dir: Path, model_registry, args, dataset=None) -> list:
    """Build image_samples list for generation, handling i2i/i2v preprocessing."""
    image_samples = []
    task = model_registry.get_task(args.diffusion_model)
    if annotations_chunk_dir.exists() and any(annotations_chunk_dir.iterdir()):
        for json_filename in sorted(os.listdir(annotations_chunk_dir)):
            if not json_filename.endswith('.json'):
                continue
            json_path = os.path.join(annotations_chunk_dir, json_filename)
            with open(json_path, 'r') as file:
                annotation = json.load(file)
            idx = annotation['id']
            image = None
            if task in ['i2i', 'i2v']:
                possible_exts = ['png', 'jpg', 'jpeg']
                for ext in possible_exts:
                    real_img_path = os.path.join(
                        f'test_data/real_images/{args.real_image_dataset_name}/{args.start_index}_{args.end_index}',
                        f"{idx}.{ext}"
                    )
                    if os.path.exists(real_img_path):
                        image = Image.open(real_img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image = np.array(image)
                        break
            # For t2i/t2v, image should be None
            if task in ['t2i', 't2v']:
                image = None
            image_samples.append({'image': image, 'prompt': annotation['description'], 'id': idx})
    else:
        # For local datasets, dataset is already sliced, so use enumerate
        # For HF datasets, use range with start_index
        if args.local_image_dir and dataset is not None:
            for index, item in enumerate(dataset):
                image = item['image']
                if isinstance(image, Image.Image):
                    if task in ['i2i', 'i2v']:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image = np.array(image)
                    else:
                        image = None
                image_samples.append({'image': image, 'id': args.start_index + index})
        elif dataset is not None:
            for idx in range(args.start_index, args.end_index + 1):
                item = dataset[idx]
                image = item['image']
                if isinstance(image, Image.Image):
                    if task in ['i2i', 'i2v']:
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image = np.array(image)
                    else:
                        image = None
                image_samples.append({'image': image, 'id': idx})
    return image_samples

def run_generation_pipeline(args, model_registry, output_dir, annotations_chunk_dir, real_images_chunk_dir, dataset=None):
    """Run the BitMind subnet pipeline for synthetic data generation."""
    image_samples = build_image_samples(annotations_chunk_dir, model_registry, args, dataset)
    pipeline = GenerationPipeline(output_dir=output_dir, model_registry=model_registry, device=f"cuda:{args.gpu_id}")
    pipeline.generate(
        image_samples=image_samples,
        tasks=[model_registry.get_task(args.diffusion_model)],
        model_names=[args.diffusion_model]
    )
    print(f"Generation complete. Outputs saved to {output_dir}")

def load_local_images(image_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
    dataset = []
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            dataset.append({'image': img, 'id': idx, 'path': path})
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return dataset

def main():
    args = parse_arguments()
    dataset = None
    if args.local_image_dir:
        dataset = load_local_images(args.local_image_dir)
        dataset_size = len(dataset)
        print(f"Loaded {dataset_size} images from {args.local_image_dir}")
        
        # Handle case where start_index is beyond dataset size
        if args.start_index >= dataset_size:
            print(f"Warning: start_index {args.start_index} >= dataset_size {dataset_size}. No images to process.")
            dataset = []
        else:
            # Adjust end_index to be within bounds
            if args.end_index is None or args.end_index >= dataset_size:
                args.end_index = dataset_size - 1
            
            # Slice the dataset
            dataset = dataset[args.start_index:args.end_index + 1]
            print(f"Sliced dataset to images {args.start_index}-{args.end_index} ({len(dataset)} images)")
        
        hf_dataset_name = args.local_image_dir
    else:
        hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"
        print(f"Loading dataset {hf_dataset_name} to determine size...")
        dataset = load_dataset(hf_dataset_name, split='train')
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size} images")
        if args.max_images is not None:
            args.end_index = args.start_index + args.max_images - 1
            if args.end_index >= dataset_size:
                args.end_index = dataset_size - 1
                print(f"Adjusted end_index to {args.end_index} (dataset size - 1)")
        elif args.end_index is None or args.end_index >= dataset_size:
            args.end_index = dataset_size - 1
            print(f"Adjusted end_index to {args.end_index} (dataset size - 1)")
    model_name = args.diffusion_model.split('/')[-1]
    data_range = f"{args.start_index}-to-{args.end_index}"
    default_name = f"{hf_dataset_name}___{data_range}___{model_name}"
    if len(default_name) > 96 and not args.output_repo_name:
        raise ValueError(
            f"Default repository name '{default_name}' exceeds HuggingFace's 96 character limit.\n"
            f"Please provide a shorter custom name using --output_repo_name.\n"
            f"Current length: {len(default_name)} characters"
        )
    hf_synthetic_images_name = (
        f"{args.hf_org}/{args.output_repo_name}"
        if args.output_repo_name
        else default_name
    )
    target_org = args.target_org if hasattr(args, 'target_org') and args.target_org else args.hf_org
    hf_annotations_name = f"{target_org}/{args.real_image_dataset_name}___annotations"
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    annotations_chunk_dir = Path(f"{annotations_dir}/{args.start_index}_{args.end_index}/")
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(annotations_chunk_dir, exist_ok=True)
    real_image_samples_dir = f'test_data/real_images/{args.real_image_dataset_name}'
    real_images_chunk_dir = Path(f'{real_image_samples_dir}/{args.start_index}_{args.end_index}/')
    model_registry = initialize_model_registry()
    task = model_registry.get_task(args.diffusion_model)
    if task in ['t2v', 'i2v']:
        synthetic_items_dir = f'test_data/synthetic_videos/{model_name}/{args.real_image_dataset_name}'
    else:
        synthetic_items_dir = f'test_data/synthetic_images/{model_name}/{args.real_image_dataset_name}'
    if task in ['i2i', 'i2v'] and args.download_real_images:
        print(f"Downloading real images for {task} from {hf_dataset_name}")
        download_real_images(dataset, args.start_index, args.end_index, real_images_chunk_dir)
    synthetic_items_chunk_dir = Path(f'{synthetic_items_dir}/{args.start_index}_{args.end_index}/')
    os.makedirs(synthetic_items_chunk_dir, exist_ok=True)
    batch_size = 32
    prompt_generator = PromptGenerator(
        vlm_name=IMAGE_ANNOTATION_MODEL,
        llm_name=TEXT_MODERATION_MODEL,
        device=f'cuda:{args.gpu_id}'
    )
    if args.download_annotations and dataset_exists_on_hf(hf_annotations_name, args.hf_token):
        print("Annotations exist on Hugging Face.")
        annotations_chunk_dir.mkdir(parents=True, exist_ok=True)
        if not annotations_chunk_dir.is_dir() or not any(annotations_chunk_dir.iterdir()):
            print(f"Downloading annotations from {hf_annotations_name} and saving annotations to {annotations_chunk_dir}.")
            all_annotations = load_dataset(hf_annotations_name, split=args.annotation_split, keep_in_memory=False)
            df_annotations = pd.DataFrame(all_annotations)
            all_annotations = None
            df_annotations['id'] = df_annotations['id'].astype(int)
            df_annotations.sort_values('id', inplace=True)
            annotations_chunk = df_annotations.iloc[args.start_index:args.end_index + 1]
            df_annotations = None
            save_as_json(annotations_chunk, annotations_chunk_dir)
            annotations_chunk = None
        else:
            print("Annotations already saved to disk.")
    elif not args.skip_generate_annotations:
        print("Generating new annotations.")
        if args.local_image_dir:
            images_chunk = dataset
        else:
            dataset = load_dataset(hf_dataset_name, split='train')
            images_chunk = slice_dataset(dataset, start_index=args.start_index, end_index=args.end_index)
        generate_and_save_annotations(images_chunk, args.start_index, hf_dataset_name, prompt_generator, annotations_chunk_dir, args, batch_size)
        prompt_generator.clear_gpu()
    if args.generate_synthetic_images:
        if not args.diffusion_model:
            raise ValueError("--diffusion_model is required when --generate_synthetic_images is specified")
        if args.local_image_dir:
            image_samples = dataset
        else:
            if task in ['t2v', 'i2v']:
                output_dir = Path(f'test_data/synthetic_videos/{model_name}/{args.real_image_dataset_name}/{args.start_index}_{args.end_index}')
            else:
                output_dir = Path(f'test_data/synthetic_images/{model_name}/{args.real_image_dataset_name}/{args.start_index}_{args.end_index}')
            output_dir.mkdir(parents=True, exist_ok=True)
            run_generation_pipeline(args, model_registry, output_dir, annotations_chunk_dir, real_images_chunk_dir, dataset)

if __name__ == "__main__":
    main()