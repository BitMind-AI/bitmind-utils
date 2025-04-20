#!/usr/bin/env python3
"""
Script to download images from a Hugging Face dataset and save them to a local directory.

Usage:
    python download_hf_dataset.py dataset_name [--output-dir OUTPUT_DIR] [--limit LIMIT] [--hf-token HF_TOKEN]

Example:
    python download_hf_dataset.py bitmind/GenImage_ADM --output-dir ./downloaded_images --limit 100 --hf-token YOUR_TOKEN
"""

import argparse
import os
import logging
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import io

def setup_logger():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_dataset_images(dataset_name, output_dir, limit=None, token=None):
    """Download images from a Hugging Face dataset and save them to a local directory."""
    logger = setup_logger()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, token=token)
        
        # Get the first split (usually 'train')
        split_name = list(dataset.keys())[0]
        split = dataset[split_name]
        
        # Limit the number of images if specified
        if limit:
            logger.info(f"Limiting to {limit} images")
            split = split.select(range(min(limit, len(split))))
        
        logger.info(f"Found {len(split)} images in dataset")
        
        # Create subdirectories based on metadata if available
        metadata_dirs = {}
        
        # Check if the dataset has any metadata columns we can use for organizing
        columns = split.column_names
        metadata_columns = [col for col in columns if col not in ['image', 'file_name']]
        
        # Download and save each image
        for i, example in enumerate(tqdm(split, desc="Downloading images")):
            # Get image data
            image = example['image']
            
            # Get filename (use index if not available)
            if 'file_name' in example:
                filename = example['file_name']
            else:
                # Extract extension from PIL image
                img = Image.open(io.BytesIO(image['bytes'])) if 'bytes' in image else image
                ext = img.format.lower() if img.format else 'jpg'
                filename = f"image_{i:05d}.{ext}"
            
            # Create subdirectory based on metadata if available
            subdir = output_dir
            if metadata_columns:
                # Use the first metadata column for subdirectory
                metadata_col = metadata_columns[0]
                metadata_val = str(example[metadata_col])
                
                if metadata_val not in metadata_dirs:
                    metadata_dirs[metadata_val] = os.path.join(output_dir, metadata_val)
                    os.makedirs(metadata_dirs[metadata_val], exist_ok=True)
                
                subdir = metadata_dirs[metadata_val]
            
            # Save the image
            output_path = os.path.join(subdir, filename)
            
            # Handle different image formats in the dataset
            if 'bytes' in image:
                with open(output_path, 'wb') as f:
                    f.write(image['bytes'])
            else:
                # If image is already a PIL Image
                image.save(output_path)
            
        logger.info(f"Successfully downloaded {len(split)} images to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download images from a Hugging Face dataset')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset (e.g., "bitmind/GenImage_ADM")')
    parser.add_argument('--output-dir', type=str, default='./downloaded_images', 
                        help='Directory to save the images (default: ./downloaded_images)')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Maximum number of images to download (default: all)')
    parser.add_argument('--hf-token', type=str, default=None, 
                        help='Hugging Face token for accessing private datasets')
    
    args = parser.parse_args()
    
    download_dataset_images(
        args.dataset_name, 
        args.output_dir, 
        args.limit, 
        args.hf_token
    )

if __name__ == "__main__":
    main()