import argparse
import logging
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
import jsonlines
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datasets import load_dataset
from utils.hugging_face_utils import (
    dataset_exists_on_hf,
    load_and_sort_dataset,
    slice_dataset,
    save_as_json,
    upload_videos_as_files
)
from utils.batch_prompt_utils import batch_process_dataset
from utils.image_utils import resize_image, resize_images_in_directory
from diffusers.utils import export_to_video
from bitmind.generation.prompt_generator import PromptGenerator
from bitmind.generation.generation_pipeline import GenerationPipeline, IMAGE_ANNOTATION_MODEL, TEXT_MODERATION_MODEL
from bitmind.generation.models import initialize_model_registry
from bitmind.transforms import apply_random_augmentations

TARGET_IMAGE_SIZE = (256, 256)
PROGRESS_INCREMENT = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generation_debug.log"),
        logging.StreamHandler()
    ]
)
logging.info("=== PYTHON LOGGER TEST: Script started ===")

def parse_arguments():
    """Parse command-line arguments for generating synthetic images and annotations.

    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login

    Do not add token as Git credential.

    Example Usage:

    Generate the first 10 mirrors of celeb-a-hq with stabilityai/stable-diffusion-xl-base-1.0
    and existing annotations from Hugging Face, and upload images to Hugging Face.
    Replace YOUR_HF_TOKEN with your actual Hugging Face API token:

    pm2 start generate_synthetic_dataset.py --name "first_ten_celebahq" --no-autorestart \
    -- --hf_org 'bitmind' --real_image_dataset_name 'celeb-a-hq' \
    --diffusion_model 'stabilityai/stable-diffusion-xl-base-1.0' --upload_synthetic_images \
    --hf_token 'YOUR_HF_TOKEN' --start_index 0 --end_index 10 --gpu_id 0

    Generate mirrors of the entire ffhq256 using stabilityai/stable-diffusion-xl-base-1.0
    and upload annotations and images to Hugging Face. Replace YOUR_HF_TOKEN with your
    actual Hugging Face API token:

    pm2 start generate_synthetic_dataset.py --name "ffhq256" --no-autorestart \
    -- --hf_org "bitmind" --real_image_dataset_name "ffhq256" \
    --diffusion_model "stabilityai/stable-diffusion-xl-base-1.0" \
    --upload_annotations --upload_synthetic_images --hf_token "YOUR_HF_TOKEN" \
    --gpu_id 0 --download_real_images

    Arguments:
        --hf_org (str): Required. Hugging Face organization name.
        --real_image_dataset_name (str): Required. Name of the real image dataset.
        --diffusion_model (str): Required. Diffusion model to use for image generation.
        --upload_annotations (bool): Optional. Flag to upload annotations to Hugging Face.
        --download_annotations (bool): Optional. Flag to download existing annotations.
        --skip_generate_annotations (bool): Optional. Flag to skip local annotation generation.
                                        Useful when local annotations exist.
        --generate_synthetic_images (bool): Optional. Flag to generate synthetic images.
        --upload_synthetic_images (bool): Optional. Flag to upload synthetic images.
        --hf_token (str): Required for interfacing with Hugging Face.
        --start_index (int): Required. Start index (inclusive) for processing the dataset.
        --end_index (int): Optional. End index (inclusive) for processing the dataset.
        --gpu_id (int): Required. Which GPU to use (check nvidia-smi -L).
        --no-resize (bool): Optional. Do not resize to target image size from BitMind constants.
        --resize_existing (bool): Optional. Resize existing image files.
        --download_real_images (bool): Optional. Download real images for i2i generation.
        --target_org (str): Optional. Target Hugging Face org for uploading annotations (default: same as --hf_org)
        --private (bool): Optional. Upload the dataset as private
        --max_images (int): Optional. Maximum number of images to annotate per dataset.
        --annotation_split (str): Optional. Which split to load from the annotation dataset (default: train)
        --test_mask_randomization (bool): Optional. Test a variety of mask randomization settings and log mask parameters/results to a JSONL file.
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
        required=True,
        help='Diffusion model to use for image generation.'
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
        '--no-resize',
        action='store_false',
        dest='resize',
        help='Do not resize to target image size from BitMind constants.'
    )
    parser.add_argument(
        '--resize_existing',
        action='store_true',
        default=False,
        required=False,
        help='Resize existing image files.'
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
            adjusted_index = index + start_index

            # Generate annotation without reloading models
            annotation = {
                "id": adjusted_index,
                "dataset": dataset_name,
                "description": prompt_generator.generate(real_image['image'], task=args.annotation_task)
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


def save_generated_items(
    json_filenames,
    annotations_dir,
    synthetic_data_generator,
    output_dir,
    real_images_dir,
    resize=True
):
    """Save generated items (images or videos) from annotations."""
    total_items = 0
    model_registry = initialize_model_registry()
    task = model_registry.get_task(synthetic_data_generator.model_name)
    modality = model_registry.get_modality(synthetic_data_generator.model_name)
    
    model_config = model_registry.get_model(synthetic_data_generator.model_name)

    for json_filename in json_filenames:
        json_path = os.path.join(annotations_dir, json_filename)
        with open(json_path, 'r') as file:
            annotation = json.load(file)
        prompt = annotation['description']
        name = annotation['id']

        # Handle i2i and i2v cases by loading source image
        image = None
        if task in ['i2i', 'i2v']:
            # Try to find the image with any common extension
            possible_exts = ['png', 'jpg', 'jpeg']
            found = False
            for ext in possible_exts:
                source_image_path = os.path.join(real_images_dir, f"{name}.{ext}")
                if os.path.exists(source_image_path):
                    image = Image.open(source_image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    found = True
                    break
            if not found:
                print(f"Source image not found for {task}: {os.path.join(real_images_dir, f'{name}.*')}")
                continue

        # Use generate_from_prompt for all tasks
        result = synthetic_data_generator.generate_from_prompt(
            prompt=prompt,
            task=task,
            image=image,
            generate_at_target_size=False
        )

        # --- DEBUG: Log result keys/type ---
        logging.info(f"Processing annotation {name}, result type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}, full result: {result}")
        if isinstance(result, dict) and 'gen_output' in result:
            logging.info(f"gen_output type: {type(result['gen_output'])}, dir: {dir(result['gen_output'])}")

        # Handle different output types
        if modality == 'video':
            filename = f"{name}.mp4"
            file_path = os.path.join(output_dir, filename)
            if 'gen_output' in result and hasattr(result['gen_output'], 'frames'):
                # Get fps from model config, default to 8 if not specified
                fps = model_config.get('save_args', {}).get('fps', 8)
                print(f"Saving video with {fps} fps based on model configuration")
                export_to_video(result['gen_output'].frames[0], file_path, fps=fps)
                total_items += 1
        else:  # image output
            filename = f"{name}.png"
            file_path = os.path.join(output_dir, filename)
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                # All image/mask saving and augmentation is now handled in the generation pipeline.
                # Only handle metadata or result processing here if needed.
                break

    return total_items


def generate_and_save_synthetic_items(
    annotations_dir,
    synthetic_data_generator,
    output_dir,
    real_images_dir,
    start_index,
    end_index,
    batch_size=16,
    resize=True
):
    """Generate and save synthetic items (images or videos) from annotations."""
    start_time = time.time()
    total_items = 0

    # Collect all valid annotation file paths first
    valid_files = []
    for json_filename in sorted(os.listdir(annotations_dir)):
        try:
            file_index = int(json_filename[:-5])
        except ValueError:
            continue
        if start_index <= file_index <= end_index:
            valid_files.append(json_filename)

    total_valid_files = len(valid_files)
    progress_interval = (
        batch_size * ceil(total_valid_files / (PROGRESS_INCREMENT * batch_size))
    )

    with torch.no_grad():
        for i in range(0, total_valid_files, batch_size):
            batch_files = valid_files[i:i+batch_size]
            total_items += save_generated_items(
                batch_files,
                annotations_dir,
                synthetic_data_generator,
                output_dir,
                real_images_dir,
                resize
            )

            if i % progress_interval == 0 or total_items >= total_valid_files:
                print(
                    f"Progress: {total_items}/{total_valid_files} items generated "
                    f"({(total_items / total_valid_files) * 100:.2f}%)"
                )

    synthetic_data_generator.clear_gpu()
    duration = time.time() - start_time
    print(f"All {total_items} synthetic items generated in {duration:.2f} seconds.")
    print(f"Mean generation time: {duration/max(total_items, 1):.2f} seconds.")


def to_python_types(obj):
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_types(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif hasattr(obj, 'item') and callable(obj.item):
        return obj.item()
    else:
        return obj


def generate_and_save_synthetic_items_with_mask_logging(
    annotations_dir,
    synthetic_data_generator,
    output_dir,
    real_images_dir,
    start_index,
    end_index,
    model_name,
    batch_size=16,
    resize=True,
    mask_param_grid=None,
    mask_jsonl_path=None,
):
    """
    Generate and save synthetic items (images or videos) from annotations, testing a variety of mask randomization settings.
    Log mask parameters and results to a JSONL file.
    Also saves the mask image as PNG if present.
    Adds debug logging for model output.
    """
    import itertools
    from tqdm import tqdm
    import logging
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    if mask_param_grid is None:
        mask_param_grid = [
            # 1. Smallest possible, single, sharp rectangle, center
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.15,
                "allow_multiple": False,
                "allowed_shapes": ["rectangle"],
                "edge_bias": 0.0,
            },
            # 2. Largest possible, single, sharp rectangle, edge
            {
                "min_size_ratio": 0.5,
                "max_size_ratio": 0.5,
                "allow_multiple": False,
                "allowed_shapes": ["rectangle"],
                "edge_bias": 1.0,
            },
            # 3. Smallest, single circle, center
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.15,
                "allow_multiple": False,
                "allowed_shapes": ["circle"],
                "edge_bias": 0.0,
            },
            # 4. Largest, single ellipse, edge
            {
                "min_size_ratio": 0.5,
                "max_size_ratio": 0.5,
                "allow_multiple": False,
                "allowed_shapes": ["ellipse"],
                "edge_bias": 1.0,
            },
            # 5. Smallest, single triangle, center
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.15,
                "allow_multiple": False,
                "allowed_shapes": ["triangle"],
                "edge_bias": 0.0,
            },
            # 6. Largest, single triangle, edge
            {
                "min_size_ratio": 0.5,
                "max_size_ratio": 0.5,
                "allow_multiple": False,
                "allowed_shapes": ["triangle"],
                "edge_bias": 1.0,
            },
            # 7. Multiple, all shapes, min size, center
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.5,
                "allow_multiple": True,
                "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"],
                "edge_bias": 0.0,
            },
            # 8. Multiple, all shapes, max size, edge
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.5,
                "allow_multiple": True,
                "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"],
                "edge_bias": 1.0,
            },
            # 9. Multiple, all shapes, random edge/center
            {
                "min_size_ratio": 0.15,
                "max_size_ratio": 0.5,
                "allow_multiple": True,
                "allowed_shapes": ["rectangle", "circle", "ellipse", "triangle"],
                "edge_bias": 0.5,
            },
        ]
    if mask_jsonl_path is None:
        mask_jsonl_path = os.path.join(str(output_dir), "mask_generation_log.jsonl")
    # Collect all valid annotation file paths first
    valid_files = []
    for json_filename in sorted(os.listdir(annotations_dir)):
        try:
            file_index = int(json_filename[:-5])
        except ValueError:
            continue
        if start_index <= file_index <= end_index:
            valid_files.append(json_filename)
    total_valid_files = len(valid_files)
    model_registry = initialize_model_registry()
    task = model_registry.get_task(model_name)
    modality = model_registry.get_modality(model_name)
    with jsonlines.open(mask_jsonl_path, mode='w') as writer:
        with torch.no_grad():
            for i in tqdm(range(0, total_valid_files, batch_size)):
                batch_files = valid_files[i:i+batch_size]
                for json_filename in batch_files:
                    json_path = os.path.join(annotations_dir, json_filename)
                    with open(json_path, 'r') as file:
                        annotation = json.load(file)
                    prompt = annotation['description']
                    name = annotation['id']
                    # For i2i/i2v, load the real image
                    image = None
                    if task == 'i2i':
                        possible_exts = ['png', 'jpg', 'jpeg']
                        for ext in possible_exts:
                            real_img_path = os.path.join(real_images_dir, f"{name}.{ext}")
                            if os.path.exists(real_img_path):
                                image = Image.open(real_img_path)
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                image = np.array(image)
                                break
                    for mask_params in mask_param_grid:
                        synthetic_data_generator.mask_params = mask_params
                        result = synthetic_data_generator.generate_from_prompt(
                            prompt=prompt,
                            task=task,
                            image=image,
                            generate_at_target_size=False,
                            mask_params=mask_params
                        )
                        # Debug logging for model output
                        print(f"Result for id {name} with mask_params {mask_params}: {result['gen_output']}")
                        if hasattr(result['gen_output'], 'images'):
                            print(f"Images: {getattr(result['gen_output'], 'images', None)}")
                        if hasattr(result['gen_output'], 'mask_image'):
                            print(f"Mask image: {getattr(result['gen_output'], 'mask_image', None)}")
                        # Save image and log mask params
                        if (
                            modality == 'image'
                            and 'gen_output' in result
                            and 'image' in result['gen_output']
                            and hasattr(result['gen_output']['image'], 'images')
                            and result['gen_output']['image'].images
                        ):
                            img = result['gen_output']['image'].images[0]
                            # Always apply base augmentation to generated image before saving
                            img_np = np.array(img)
                            img_aug, _, _ = apply_random_augmentations(img_np, (256, 256), level_probs={0: 1.0})
                            img = Image.fromarray(img_aug)
                            # Save augmented image
                            out_name = f"{name}_masktest_{hash(str(mask_params)) & 0xFFFF}.png"
                            out_path = os.path.join(output_dir, out_name)
                            try:
                                img.save(out_path)
                                print(f"Saved image: {out_path}")
                            except Exception as e:
                                print(f"Failed to save image {out_path}: {e}")
                                logging.error(f"Failed to save image {out_path}: {e}")
                            # Always apply base augmentation to mask before saving
                            mask_img = result['gen_output'].get('mask_image', None)
                            if mask_img is not None:
                                mask_np = np.array(mask_img)
                                if mask_np.size == 0:
                                    print(f"Warning: Mask for id {name} is empty, skipping mask save.")
                                else:
                                    if mask_np.ndim == 1:
                                        side = int(np.sqrt(mask_np.shape[0]))
                                        if side * side != mask_np.shape[0]:
                                            print(f"Warning: Mask for id {name} shape {mask_np.shape} is not square, skipping mask save.")
                                        else:
                                            mask_np = mask_np.reshape((side, side))
                                    if mask_np.ndim == 3 and mask_np.shape[2] == 3:
                                        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
                                    if mask_np.ndim == 2:
                                        mask_np = mask_np[..., None]  # Make it (H, W, 1)
                                    if mask_np.ndim != 3:
                                        print(f"ERROR: Mask for id {name} is not 3D after all conversions, shape: {mask_np.shape}")
                                    else:
                                        mask_aug, _, _ = apply_random_augmentations(mask_np, (256, 256), level_probs={0: 1.0})
                                        if mask_aug.ndim == 3 and mask_aug.shape[2] == 1:
                                            mask_aug = np.squeeze(mask_aug, axis=2)
                                        # Save as PNG
                                        mask_img_aug = Image.fromarray(mask_aug)
                                        mask_out_path = out_path.replace('.png', '_mask.png')
                                        mask_img_aug.save(mask_out_path)
                                        print(f"Saved mask: {mask_out_path}")
                                        # Save as .npy
                                        mask_npy_out_path = mask_out_path.replace('.png', '.npy')
                                        np.save(mask_npy_out_path, mask_aug)
                                print(f"Saved mask (npy): {mask_npy_out_path}")
                            # Save original (real) image with base augmentations for comparison
                            if image is not None:
                                real_img = image
                                if isinstance(real_img, np.ndarray):
                                    real_img = Image.fromarray(real_img)
                                real_img_aug = real_img.resize((TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]), Image.LANCZOS)
                                real_img_np = np.array(real_img_aug)
                                real_img_aug_np, _, _ = apply_random_augmentations(real_img_np, (256, 256), level_probs={0: 1.0})
                                real_img_aug_final = Image.fromarray(real_img_aug_np)
                                real_img_out_path = os.path.join(output_dir, f"{name}_real_aug.png")
                                real_img_aug_final.save(real_img_out_path)
                                print(f"Saved real augmented image: {real_img_out_path}")
                            writer.write(to_python_types({
                                "id": name,
                                "mask_params": mask_params,
                                "mask_metadata": result['gen_output'].get('mask_metadata', None),
                                "output_path": out_path,
                                "mask_output_path": mask_out_path if mask_img is not None else None
                            }))
                        else:
                            print(f"No image generated for id {name} with mask_params {mask_params}. Skipping.")
                            logging.warning(f"No image generated for id {name} with mask_params {mask_params}. Skipping.")
    synthetic_data_generator.clear_gpu()


def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"

    # Load the dataset first to determine its size
    print(f"Loading dataset {hf_dataset_name} to determine size...")
    dataset = load_dataset(hf_dataset_name, split='train')
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} images")

    # Set end_index based on max_images if provided
    if args.max_images is not None:
        args.end_index = args.start_index + args.max_images - 1
        if args.end_index >= dataset_size:
            args.end_index = dataset_size - 1
            print(f"Adjusted end_index to {args.end_index} (dataset size - 1)")
    elif args.end_index is None or args.end_index >= dataset_size:
        args.end_index = dataset_size - 1
        print(f"Adjusted end_index to {args.end_index} (dataset size - 1)")
        
    # Create default name using full model name
    model_name = args.diffusion_model.split('/')[-1]
    data_range = f"{args.start_index}-to-{args.end_index}"
    default_name = f"{hf_dataset_name}___{data_range}___{model_name}"

    # Check if default name is too long
    if len(default_name) > 96 and not args.output_repo_name:
        raise ValueError(
            f"Default repository name '{default_name}' exceeds HuggingFace's 96 character limit.\n"
            f"Please provide a shorter custom name using --output_repo_name.\n"
            f"Current length: {len(default_name)} characters"
        )

    # Use custom name if provided, otherwise use default
    hf_synthetic_images_name = (
        f"{args.hf_org}/{args.output_repo_name}"
        if args.output_repo_name
        else default_name
    )

    # Use target_org if provided, else default to hf_org
    target_org = args.target_org if hasattr(args, 'target_org') and args.target_org else args.hf_org
    hf_annotations_name = f"{target_org}/{args.real_image_dataset_name}___annotations"
    annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
    annotations_chunk_dir = Path(
        f"{annotations_dir}/{args.start_index}_{args.end_index}/"
    )
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(annotations_chunk_dir, exist_ok=True)

    real_image_samples_dir = f'test_data/real_images/{args.real_image_dataset_name}'
    real_images_chunk_dir = Path(
        f'{real_image_samples_dir}/{args.start_index}_{args.end_index}/'
    )

    model_registry = initialize_model_registry()
    task = model_registry.get_task(args.diffusion_model)
    if task in ['t2v', 'i2v']:
        synthetic_items_dir = f'test_data/synthetic_videos/{model_name}/{args.real_image_dataset_name}'
    else:
        synthetic_items_dir = f'test_data/synthetic_images/{model_name}/{args.real_image_dataset_name}'

    # Download real images for i2i and i2v tasks if requested
    if task in ['i2i', 'i2v'] and args.download_real_images:
        print(f"Downloading real images for {task} from {hf_dataset_name}")
        download_real_images(
            dataset,
            args.start_index,
            args.end_index,
            real_images_chunk_dir
        )

    synthetic_items_chunk_dir = Path(
        f'{synthetic_items_dir}/{args.start_index}_{args.end_index}/'
    )
    os.makedirs(synthetic_items_chunk_dir, exist_ok=True)

    batch_size = 32
    # Initialize the generators
    prompt_generator = PromptGenerator(
        vlm_name=IMAGE_ANNOTATION_MODEL,
        llm_name=TEXT_MODERATION_MODEL,
        device=f'cuda:{args.gpu_id}'  # Use the specified GPU
    )

    synthetic_image_generator = None

    # Generate or download annotations to local storage.
    if args.download_annotations and dataset_exists_on_hf(
        hf_annotations_name,
        args.hf_token
    ):
        print("Annotations exist on Hugging Face.")
        # Check if the annotations are already saved locally

        annotations_chunk_dir.mkdir(parents=True, exist_ok=True)
        if not annotations_chunk_dir.is_dir() or not any(
            annotations_chunk_dir.iterdir()
        ):
            print(
                f"Downloading annotations from {hf_annotations_name} and saving "
                f"annotations to {annotations_chunk_dir}."
            )
            # Download annotations from Hugging Face
            all_annotations = load_dataset(
                hf_annotations_name,
                split=args.annotation_split,
                keep_in_memory=False
            )
            df_annotations = pd.DataFrame(all_annotations)
            all_annotations = None
            # Ensure the index is of integer type and sort by it
            df_annotations['id'] = df_annotations['id'].astype(int)
            df_annotations.sort_values('id', inplace=True)
            # Slice specified chunk
            annotations_chunk = df_annotations.iloc[args.start_index:args.end_index + 1]
            df_annotations = None
            # Save the chunk as JSON files on disk
            save_as_json(annotations_chunk, annotations_chunk_dir)
            annotations_chunk = None
        else:
            print("Annotations already saved to disk.")
    elif not args.skip_generate_annotations:
        print("Generating new annotations.")
        dataset = load_dataset(hf_dataset_name, split='train')
        images_chunk = slice_dataset(
            dataset,
            start_index=args.start_index,
            end_index=args.end_index
        )
        dataset = None

        # Use the batch processing utility instead of the old function
        batch_process_dataset(
            images_chunk,
            args.start_index,
            hf_dataset_name,
            prompt_generator,
            annotations_chunk_dir,
            batch_size=batch_size,
            annotation_task=args.annotation_task
        )

        prompt_generator.clear_gpu()
        images_chunk = None  # Free up memory

    # Generate synthetic items to local storage using BitMind subnet pipeline
    if args.generate_synthetic_images:
        # Prepare image_samples from annotation chunk (or real images for i2i/i2v)
        # We'll use the annotation chunk if it exists, else fallback to all_images
        annotations_dir = f'test_data/annotations/{args.real_image_dataset_name}'
        annotations_chunk_dir = Path(
            f"{annotations_dir}/{args.start_index}_{args.end_index}/"
        )
        image_samples = []
        if annotations_chunk_dir.exists() and any(annotations_chunk_dir.iterdir()):
            # Use annotation chunk to get prompts and ids
            for json_filename in sorted(os.listdir(annotations_chunk_dir)):
                if not json_filename.endswith('.json'):
                    continue
                json_path = os.path.join(annotations_chunk_dir, json_filename)
                with open(json_path, 'r') as file:
                    annotation = json.load(file)
                idx = annotation['id']
                # For i2i/i2v, load the real image
                image = None
                if model_registry.get_task(args.diffusion_model) in ['i2i', 'i2v']:
                    possible_exts = ['png', 'jpg', 'jpeg']
                    for ext in possible_exts:
                        real_img_path = os.path.join(f'test_data/real_images/{args.real_image_dataset_name}/{args.start_index}_{args.end_index}', f"{idx}.{ext}")
                        if os.path.exists(real_img_path):
                            image = Image.open(real_img_path)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            image = np.array(image)
                            break
                image_samples.append({'image': image, 'prompt': annotation['description'], 'id': idx})
        else:
            # Fallback: use all_images
            for idx in range(args.start_index, args.end_index + 1):
                item = dataset[idx]
                image = item['image']
                if isinstance(image, Image.Image):
                    image = np.array(image)
                image_samples.append({'image': image, 'id': idx})

        # Initialize model registry and pipeline
        output_dir = Path(f'test_data/synthetic_images/{model_name}/{args.real_image_dataset_name}/{args.start_index}_{args.end_index}')
        output_dir.mkdir(parents=True, exist_ok=True)
        model_registry = initialize_model_registry()
        pipeline = GenerationPipeline(output_dir=output_dir, model_registry=model_registry, device=f"cuda:{args.gpu_id}")

        # Run generation for the specified model and all supported tasks
        print(f"Generating synthetic data using BitMind subnet pipeline for model: {args.diffusion_model}")
        if getattr(args, 'test_mask_randomization', False):
            print("Testing mask randomization and logging mask parameters/results to JSONL...")
            generate_and_save_synthetic_items_with_mask_logging(
                annotations_chunk_dir,
                pipeline,
                output_dir,
                real_images_chunk_dir,
                args.start_index,
                args.end_index,
                args.diffusion_model,
                batch_size=batch_size,
                resize=args.resize,
            )
            print(f"Mask randomization test complete. See mask_generation_log.jsonl in {output_dir}")
        else:
            pipeline.generate(
                image_samples=image_samples,
                tasks=[model_registry.get_task(args.diffusion_model)],
                model_names=[args.diffusion_model]
            )
            print(f"Generation complete. Outputs saved to {output_dir}")

    if args.resize_existing:
        print(f"Resizing images in {synthetic_items_chunk_dir}.")
        resize_images_in_directory(synthetic_items_chunk_dir)
        hf_synthetic_images_name += f"___{TARGET_IMAGE_SIZE[0]}"
        print(f"Done resizing existing images.")


if __name__ == "__main__":
    main()