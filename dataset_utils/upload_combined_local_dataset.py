#!/usr/bin/env python3
"""
Combine all local split directories for a given model and dataset, and upload the combined dataset to Hugging Face.
Supports both images and videos.

Usage:
    python upload_combined_local_dataset.py <hf_org> <dataset_name> <model_name> <hf_token> [--media_type image|video] [--private] [--base_dir BASE_DIR]

Example:
    python upload_combined_local_dataset.py bitmind celeb-a-hq stable-diffusion-xl-base-1.0 <YOUR_HF_TOKEN> --media_type image --private
"""
import argparse
import os
import glob
from datasets import Dataset, Features, Image as DsImage
from huggingface_hub import HfApi
import shutil
import time

DATASET_ALIASES = {
    "bm-real": "bm",
    "celeb-a-hq": "celeb",
    "ffhq-256": "ffhq",
    "MS-COCO-unique-256": "coco",
    "AFHQ": "afhq",
    "lfw": "lfw",
    "caltech-256": "c256",
    "caltech-101": "c101",
    "dtd": "dtd",
    "idoc-mugshots-images": "mug"
}

def find_local_split_dirs(base_dir, media_type, model_name, dataset_name):
    if media_type == 'video':
        root = os.path.join(base_dir, 'synthetic_videos', model_name, dataset_name)
    else:
        root = os.path.join(base_dir, 'synthetic_images', model_name, dataset_name)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such directory: {root}")
    # Find all subdirectories (splits)
    split_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    split_dirs.sort()
    return split_dirs

def collect_files(split_dirs, extensions):
    files = []
    for d in split_dirs:
        for ext in extensions:
            files.extend(glob.glob(os.path.join(d, f'*.{ext}')))
            files.extend(glob.glob(os.path.join(d, f'*.{ext.upper()}')))
    files.sort()
    return files

def create_image_dataset_from_files(image_files):
    # Create dataset with image column
    dataset = Dataset.from_dict({'image': image_files})
    dataset = dataset.cast_column('image', DsImage())
    return dataset

def upload_image_dataset(dataset, dataset_name, token, private):
    print(f"Uploading image dataset to {dataset_name} ...")
    dataset.push_to_hub(dataset_name, token=token, private=private)
    print(f"Successfully uploaded image dataset to {dataset_name}")
    print(f"View at: https://huggingface.co/datasets/{dataset_name}")

def upload_video_files(video_files, dataset_name, token, private):
    print(f"Uploading {len(video_files)} video files to {dataset_name} ...")
    api = HfApi(token=token)
    # Create the dataset repo if it doesn't exist
    api.create_repo(repo_id=dataset_name, repo_type="dataset", private=private, exist_ok=True)
    for i, video_path in enumerate(video_files):
        fname = os.path.basename(video_path)
        print(f"[{i+1}/{len(video_files)}] Uploading {fname} ...")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                api.upload_file(
                    path_or_fileobj=video_path,
                    path_in_repo=fname,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=token
                )
                break
            except Exception as e:
                print(f"Upload failed for {fname}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"Giving up on {fname}")
        time.sleep(1)
    print(f"Finished uploading all videos to {dataset_name}")
    print(f"View at: https://huggingface.co/datasets/{dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='Upload combined local dataset to Hugging Face (images or videos)')
    parser.add_argument('hf_org', type=str, help='Hugging Face organization name')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('model_name', type=str, help='Name of the diffusion model')
    parser.add_argument('hf_token', type=str, help='Hugging Face token')
    parser.add_argument('--media_type', type=str, default='image', choices=['image', 'video'], help='Type of media to upload (image or video)')
    parser.add_argument('--private', action='store_true', help='Make the dataset private')
    parser.add_argument('--base_dir', type=str, default='test_data', help='Base directory for synthetic data (default: test_data)')
    args = parser.parse_args()

    dataset_alias = DATASET_ALIASES.get(args.dataset_name, args.dataset_name)
    model_name = args.model_name.split('/')[-1]
    combined_dataset_name = f"{args.hf_org}/{dataset_alias}_{model_name}"

    split_dirs = find_local_split_dirs(args.base_dir, args.media_type, model_name, dataset_alias)
    if not split_dirs:
        print(f"No split directories found for {args.media_type} {model_name}/{dataset_alias}")
        return

    if args.media_type == 'image':
        image_files = collect_files(split_dirs, ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp'])
        if not image_files:
            print(f"No image files found in splits.")
            return
        print(f"Found {len(image_files)} images in {len(split_dirs)} splits.")
        dataset = create_image_dataset_from_files(image_files)
        upload_image_dataset(dataset, combined_dataset_name, args.hf_token, args.private)
    else:
        video_files = collect_files(split_dirs, ['mp4'])
        if not video_files:
            print(f"No video files found in splits.")
            return
        print(f"Found {len(video_files)} videos in {len(split_dirs)} splits.")
        upload_video_files(video_files, combined_dataset_name, args.hf_token, args.private)

if __name__ == "__main__":
    main() 