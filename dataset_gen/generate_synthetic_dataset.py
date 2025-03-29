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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from datasets import load_dataset

from synthetic_data_generator import SyntheticDataGenerator
from prompt_generator import PromptGenerator
from base_miner.datasets import ImageDataset
from bitmind.validator.config import (
    TARGET_IMAGE_SIZE,
    IMAGE_ANNOTATION_MODEL,
    TEXT_MODERATION_MODEL,
    get_task,
    get_modality
)
from utils.hugging_face_utils import (
    dataset_exists_on_hf,
    load_and_sort_dataset,
    upload_to_huggingface,
    slice_dataset,
    save_as_json
)
from utils.batch_prompt_utils import batch_process_dataset
from utils.image_utils import resize_image, resize_images_in_directory

PROGRESS_INCREMENT = 10


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
        '--upload_annotations',
        action='store_true',
        default=False,
        help='Upload annotations to Hugging Face.'
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
        '--upload_synthetic_images',
        action='store_true',
        default=False,
        help='Upload synthetic images to Hugging Face.'
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
    return parser.parse_args()


def download_real_images(dataset, start_index, end_index, output_dir, resize=True):
    """Download and optionally resize real images from the dataset.
    
    Args:
        dataset: The source dataset containing images
        start_index: Starting index for processing
        end_index: Ending index for processing
        output_dir: Directory to save the images
        resize: Whether to resize images to TARGET_IMAGE_SIZE
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

            # Resize if requested
            if resize:
                image = resize_image(
                    image,
                    TARGET_IMAGE_SIZE[0],
                    TARGET_IMAGE_SIZE[1]
                )

            # Save the image
            image_path = os.path.join(output_dir, f"{start_index + idx}.png")
            image.save(image_path)
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
                "description": prompt_generator.generate(real_image['image'])
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
    task = get_task(synthetic_data_generator.model_name)
    modality = get_modality(synthetic_data_generator.model_name)

    for json_filename in json_filenames:
        json_path = os.path.join(annotations_dir, json_filename)
        with open(json_path, 'r') as file:
            annotation = json.load(file)
        prompt = annotation['description']
        name = annotation['id']

        # Handle i2i case by loading source image
        image = None
        if task == 'i2i':
            source_image_path = os.path.join(real_images_dir, f"{name}.png")
            if not os.path.exists(source_image_path):
                print(f"Source image not found for i2i: {source_image_path}")
                continue
            image = Image.open(source_image_path)

        # Use generate_from_prompt for all tasks
        result = synthetic_data_generator.generate_from_prompt(
            prompt=prompt,
            task=task,
            image=image,
            generate_at_target_size=False
        )

        # Handle different output types
        if modality == 'video':
            filename = f"{name}.mp4"
            file_path = os.path.join(output_dir, filename)
            if 'gen_output' in result and hasattr(result['gen_output'], 'frames'):
                export_to_video(result['gen_output'].frames[0], file_path, fps=30)
                total_items += 1
        else:  # image output
            filename = f"{name}.png"
            file_path = os.path.join(output_dir, filename)
            if 'gen_output' in result and hasattr(result['gen_output'], 'images'):
                image = result['gen_output'].images[0]
                if resize and modality == 'image':
                    image = resize_image(
                        image,
                        TARGET_IMAGE_SIZE[0],
                        TARGET_IMAGE_SIZE[1]
                    )
                image.save(file_path)
                total_items += 1

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


def main():
    args = parse_arguments()
    hf_dataset_name = f"{args.hf_org}/{args.real_image_dataset_name}"

    # Load the dataset first to determine its size
    print(f"Loading dataset {hf_dataset_name} to determine size...")
    all_images = ImageDataset(
        huggingface_dataset_path=hf_dataset_name,
        huggingface_dataset_split='train'
    )
    dataset_size = len(all_images.dataset)
    print(f"Dataset size: {dataset_size} images")

    # Adjust end_index if it exceeds dataset size
    if args.end_index is None or args.end_index >= dataset_size:
        args.end_index = dataset_size - 1
        print(f"Adjusted end_index to {args.end_index} (dataset size - 1)")

    data_range = f"{args.start_index}-to-{args.end_index}"
    hf_annotations_name = f"{hf_dataset_name}___annotations"
    model_name = args.diffusion_model.split('/')[-1]
    hf_synthetic_images_name = f"{hf_dataset_name}___{data_range}___{model_name}"
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

    task = get_task(args.diffusion_model)
    if task == 't2v':
        synthetic_items_dir = f'test_data/synthetic_videos/{args.real_image_dataset_name}'
    else:
        if task == 'i2i' and args.download_real_images:
            print(f"Downloading and resizing real images for i2i from {hf_dataset_name}")
            download_real_images(
                all_images.dataset,
                args.start_index,
                args.end_index,
                real_images_chunk_dir,
                resize=args.resize
            )
        synthetic_items_dir = f'test_data/synthetic_images/{args.real_image_dataset_name}'

    synthetic_items_chunk_dir = Path(
        f'{synthetic_items_dir}/{args.start_index}_{args.end_index}/'
    )
    os.makedirs(synthetic_items_chunk_dir, exist_ok=True)

    batch_size = 16
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
                split='train',
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
        all_images = ImageDataset(hf_dataset_name, 'train')
        images_chunk = slice_dataset(
            all_images.dataset,
            start_index=args.start_index,
            end_index=args.end_index
        )
        all_images = None

        # Use the batch processing utility instead of the old function
        batch_process_dataset(
            images_chunk,
            args.start_index,
            hf_dataset_name,
            prompt_generator,
            annotations_chunk_dir,
            batch_size=batch_size
        )

        prompt_generator.clear_gpu()
        images_chunk = None  # Free up memory

    # Upload to Hugging Face
    if args.upload_annotations and args.hf_token:
        start_time = time.time()
        print("Uploading annotations to HF.")
        print("Loading annotations dataset.")
        annotations_dataset = load_and_sort_dataset(annotations_chunk_dir, 'json')
        print(
            "Uploading annotations of " + args.real_image_dataset_name +
            " to Hugging Face."
        )
        upload_to_huggingface(
            annotations_dataset,
            hf_annotations_name,
            args.hf_token
        )
        print(
            f"Annotations uploaded to Hugging Face in {time.time() - start_time:.2f} "
            "seconds."
        )

    # Generate synthetic items to local storage.
    if args.generate_synthetic_images:
        # Initialize the synthetic data generator with the specified diffusion model
        synthetic_data_generator = SyntheticDataGenerator(
            model_name=args.diffusion_model,
            use_random_model=False,
            prompt_type='none',  # We'll provide prompts directly
            device=f'cuda:{args.gpu_id}'
        )

        synthetic_items_chunk_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating and saving items to {synthetic_items_chunk_dir}.")
        generate_and_save_synthetic_items(
            annotations_chunk_dir,
            synthetic_data_generator,
            synthetic_items_chunk_dir,
            real_images_chunk_dir,
            args.start_index,
            args.end_index,
            batch_size=batch_size,
            resize=args.resize
        )

        synthetic_data_generator.clear_gpu()

    if args.resize_existing:
        print(f"Resizing images in {synthetic_items_chunk_dir}.")
        resize_images_in_directory(synthetic_items_chunk_dir)
        hf_synthetic_images_name += f"___{TARGET_IMAGE_SIZE[0]}"
        print(f"Done resizing existing images.")

    if args.upload_synthetic_images and args.hf_token:
        start_time = time.time()
        print("Loading synthetic image dataset.")
        synthetic_image_dataset = load_and_sort_dataset(
            synthetic_items_chunk_dir,
            'image'
        )
        print(
            "Uploading synthetic image mirrors of " + args.real_image_dataset_name +
            " to Hugging Face."
        )
        upload_to_huggingface(
            synthetic_image_dataset,
            hf_synthetic_images_name,
            args.hf_token
        )
        print(
            f"Synthetic images uploaded in {time.time() - start_time:.2f} seconds."
        )


if __name__ == "__main__":
    main()