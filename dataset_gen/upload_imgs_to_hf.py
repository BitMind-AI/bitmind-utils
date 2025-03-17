#!/usr/bin/env python3
"""
Script to upload a directory of images to Hugging Face as a dataset.

Usage:
    python upload_imgs_to_hf.py input_dir dataset_name --hf-token YOUR_TOKEN [--private] [--batch-size BATCH_SIZE]

Examples:
    # Upload to your personal account
    python upload_imgs_to_hf.py ./my_images username/my_dataset --hf-token YOUR_TOKEN --private
    
    # Upload to an organization
    python upload_imgs_to_hf.py ./my_images organization-name/my_dataset --hf-token YOUR_TOKEN --private
"""

import argparse
import os
import logging
from tqdm import tqdm
from datasets import Dataset, Features, Image as DsImage
from huggingface_hub import HfApi
import glob

def setup_logger():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_images(directory):
    """Find all image files in a directory (including subdirectories)."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(directory, '**', ext.upper()), recursive=True))
    
    return sorted(image_files)

def create_dataset_from_directory(directory, batch_size=1000):
    """Create a Hugging Face dataset from a directory of images."""
    logger = setup_logger()
    
    # Find all image files
    logger.info(f"Scanning for images in {directory}")
    image_files = find_images(directory)
    
    if not image_files:
        logger.error(f"No image files found in {directory}")
        return None
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process images in batches to avoid memory issues with large datasets
    all_examples = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_examples = []
        
        for img_path in tqdm(batch_files, desc=f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}"):
            # Create example with only the image path
            example = {
                'image_path': img_path
            }
            
            batch_examples.append(example)
        
        all_examples.extend(batch_examples)
    
    # Create dataset
    logger.info("Creating dataset with only images (no metadata)")
    
    # First create dataset with image paths
    dataset = Dataset.from_dict({
        'image': [ex['image_path'] for ex in all_examples]
    })
    
    # Then cast the image column to Image type
    dataset = dataset.cast_column('image', DsImage())
    
    return dataset

def upload_dataset_to_hf(dataset, dataset_name, token, private=True):
    """Upload a dataset to Hugging Face."""
    logger = setup_logger()
    
    logger.info(f"Uploading dataset to {dataset_name}")
    
    try:
        # Check if token is valid and has access to the organization if specified
        api = HfApi(token=token)
        user_info = api.whoami()
        
        # Check if dataset_name includes an organization
        if '/' in dataset_name:
            org_name = dataset_name.split('/')[0]
            # Check if user has access to the organization
            orgs = [org['name'] for org in user_info.get('orgs', [])]
            if org_name not in orgs and org_name != user_info['name']:
                logger.warning(f"You don't appear to be a member of the organization '{org_name}'. "
                              f"Make sure you have the necessary permissions.")
        
        # Push to hub
        dataset.push_to_hub(
            dataset_name,
            token=token,
            private=private
        )
        
        logger.info(f"Successfully uploaded dataset to {dataset_name}")
        logger.info(f"View your dataset at: https://huggingface.co/datasets/{dataset_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload a directory of images to Hugging Face')
    parser.add_argument('input_dir', type=str, help='Directory containing images to upload')
    parser.add_argument('dataset_name', type=str, 
                        help='Name for the dataset on Hugging Face (e.g., "username/dataset_name" or "organization/dataset_name")')
    parser.add_argument('--hf-token', type=str, required=True, help='Hugging Face token')
    parser.add_argument('--private', action='store_true', help='Make the dataset private (default: False)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing images (default: 1000)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    # Validate dataset name format
    if '/' not in args.dataset_name:
        print(f"Warning: Dataset name should include a namespace (username or organization name) followed by '/'")
        print(f"Example: 'username/dataset_name' or 'organization/dataset_name'")
        confirm = input("Do you want to continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    # Create dataset from directory
    dataset = create_dataset_from_directory(args.input_dir, args.batch_size)
    
    if dataset is None:
        return
    
    # Upload dataset to Hugging Face
    upload_dataset_to_hf(dataset, args.dataset_name, args.hf_token, args.private)

if __name__ == "__main__":
    main()