"""
Script for processing and reuploading split zip archives as separate datasets.

sudo apt-get install p7zip-full

Sample PM2 commands:

Process all directories:
pm2 start combine_split_zip.py --name "process-genimage" --no-autorestart -- jzousz/GenImage bitmind --hf-token YOUR_TOKEN

Process specific directories:
pm2 start combine_split_zip.py --name "process-adm" --no-autorestart -- jzousz/GenImage bitmind --directories ADM --hf-token YOUR_TOKEN

Process with limits:
pm2 start combine_split_zip.py --name "process-adm-sample" --no-autorestart -- jzousz/GenImage bitmind --directories ADM --hf-token YOUR_TOKEN --sample-archives 7 --max-images 26000

Inspect mode (keeps temp files):
pm2 start combine_split_zip.py --name "process-adm-inspect" --no-autorestart -- jzousz/GenImage bitmind --directories ADM --hf-token YOUR_TOKEN --inspect

Monitor logs:
pm2 logs process-genimage

Stop process:
pm2 stop process-genimage
"""

import argparse
import glob
import io
import json
import logging
import os
import shutil
import tempfile
import time
import threading
from pathlib import Path
from typing import List, Optional

import requests
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm
import zipfile

def setup_logger():
    """Set up basic logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('logs', 'processing.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def list_hf_files(repo_id: str, token: str, directory: str = None) -> List[str]:
    """List files in a Hugging Face dataset repository."""
    try:
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        if directory:
            # Filter files to only include those in the specified directory
            files = [f for f in files if f.startswith(f"{directory}/")]
        return files
    except Exception as e:
        logging.error(f"Error listing files from {repo_id}: {str(e)}")
        return []

def download_hf_file(repo_id: str, file_path: str, token: str, output_dir: str) -> str:
    """Download a file from Hugging Face."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            token=token,
            local_dir=output_dir
        )
        return local_path
    except Exception as e:
        logging.error(f"Error downloading {file_path} from {repo_id}: {str(e)}")
        raise

def find_split_archives_on_hf(repo_id: str, directory: str, token: str) -> List[str]:
    """Find all split ZIP archives in the HF directory."""
    files = list_hf_files(repo_id, token, directory)
    logging.info(f"Files in {directory}: {files}")
    
    # Find all .z01 files
    z01_files = [f for f in files if f.endswith('.z01') or f.endswith('.Z01')]
    valid_archives = []
    
    for z01_file in z01_files:
        # Extract the base name without extension
        base_path = z01_file[:-4]  # Remove .z01
        
        # For a file like "ADM/imagenet_ai_0508_adm.z01", we want "ADM/imagenet_ai_0508_adm.zip"
        zip_file = base_path + '.zip'  # Just replace .z01 with .zip
        
        logging.info(f"Checking for ZIP file: {zip_file}")
        
        # Check if the .zip file exists
        if zip_file not in files:
            logging.warning(f"Found {z01_file} but missing corresponding {zip_file}")
            continue
            
        # Verify all parts exist
        is_complete = True
        part_num = 1
        while True:
            part_file = f"{base_path}.z{part_num:02d}"
            if part_file not in files:
                if part_num == 1:
                    is_complete = False
                break
            part_num += 1
            
        if is_complete:
            valid_archives.append(z01_file)
            logging.info(f"Found complete archive: {z01_file}")
        else:
            logging.warning(f"Incomplete archive set found for {os.path.basename(base_path)}")
    
    return valid_archives

def get_processed_files(directory):
    """Get list of already processed files from status file."""
    status_file = os.path.join('logs', f'{directory}_processed.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return set(json.load(f))
    return set()

def update_processed_files(directory, filename):
    """Update the status file with processed file."""
    status_file = os.path.join('logs', f'{directory}_processed.json')
    processed = get_processed_files(directory)
    processed.add(filename)
    with open(status_file, 'w') as f:
        json.dump(list(processed), f)

def animated_extraction(stop_event, file_name):
    """Display an animated progress indicator during extraction."""
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    
    # Create a progress bar that doesn't show percentage
    with tqdm(desc=f"Extracting {file_name}", bar_format='{desc}: {bar}', leave=False) as pbar:
        while not stop_event.is_set():
            pbar.set_postfix_str(spinner[i % len(spinner)])
            pbar.update(0)  # Force refresh
            i += 1
            time.sleep(0.1)

def process_directory(source_repo, directory, target_org, hf_token, max_images=None, sample_archives=None):
    """Process a single directory of images from Hugging Face."""
    logging.info(f"\nProcessing directory: {directory}")
    
    # Find all split archives on HF
    split_archives = find_split_archives_on_hf(source_repo, directory, hf_token)
    
    if not split_archives:
        logging.warning(f"No complete ZIP archives found in {directory}")
        logging.warning(f"No valid images found in {directory}. Skipping...")
        return
    
    logging.info(f"Before sampling: Found {len(split_archives)} archives in {directory}")
    if sample_archives:
        logging.info(f"Sample archives parameter is set to {sample_archives}")
    
    # Sample only a subset of archives if requested - take the first N instead of random
    if sample_archives and sample_archives < len(split_archives):
        logging.info(f"Taking first {sample_archives} archives out of {len(split_archives)}")
        split_archives = split_archives[:sample_archives]
    else:
        logging.info(f"Not sampling: sample_archives={sample_archives}, len(split_archives)={len(split_archives)}")
    
    images = []
    temp_dir = None
    
    # Create fixed extraction directory at the same level as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extract_dir = os.path.join(script_dir, "extracted_images")
    os.makedirs(extract_dir, exist_ok=True)
    
    processed_files = get_processed_files(directory)
    
    try:
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Print the directory paths
        logging.info(f"Download directory: {temp_dir}")
        logging.info(f"Extract directory: {extract_dir}")
        
        for archive in split_archives:
            # Stop if we've reached the maximum number of images
            if max_images and len(images) >= max_images:
                logging.info(f"Reached maximum number of images ({max_images}). Stopping.")
                break
                
            try:
                # Get the full base name without the .z01 extension
                base_name = archive[:-4]  # Remove .z01
                
                # Extract just the filename part for display
                file_name = os.path.basename(base_name)
                
                # Create a subdirectory for this specific archive
                archive_extract_dir = os.path.join(extract_dir, file_name)
                os.makedirs(archive_extract_dir, exist_ok=True)
                
                # Skip if archive was already processed
                if file_name in processed_files:
                    logging.info(f"Skipping already processed archive: {file_name}")
                    continue
                
                # Download all parts and the zip file
                archive_dir = os.path.dirname(archive)
                all_files = list_hf_files(source_repo, hf_token, archive_dir)
                
                # Get all files that match the base name pattern
                related_files = []
                for file in all_files:
                    if file.startswith(base_name[:-3]) and (file.endswith('.zip') or '.z' in file):
                        related_files.append(file)
                
                # Download all related files
                local_files = []
                for file in tqdm(related_files, desc=f"Downloading {file_name} parts"):
                    local_path = download_hf_file(source_repo, file, hf_token, temp_dir)
                    local_files.append(local_path)
                
                # Find the first part (.z01 file)
                z01_file = next((f for f in local_files if f.endswith('.z01')), None)
                if not z01_file:
                    logging.error(f"Could not find .z01 file for {file_name}")
                    continue
                
                # Use 7z to extract the files directly
                logging.info(f"Extracting archive: {file_name}")
                
                # Check if 7z is installed
                try:
                    import subprocess
                    
                    # Simple logging for extraction progress
                    logging.info(f"Starting extraction of {file_name}...")
                    start_time = time.time()

                    # Use 7z to extract the files
                    cmd = ["7z", "x", z01_file, f"-o{archive_extract_dir}"]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    elapsed = time.time() - start_time
                    logging.info(f"Completed extraction of {file_name} in {elapsed:.1f} seconds")

                    if result.returncode != 0:
                        logging.error(f"Error extracting with 7z: {result.stderr}")
                        continue
                        
                    logging.info(f"Successfully extracted files to {archive_extract_dir}")
                    
                    # Process all images in the extracted directory
                    image_files = []
                    for root, _, files in os.walk(archive_extract_dir):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                image_files.append(os.path.join(root, file))
                    
                    logging.info(f"Found {len(image_files)} images in {file_name}")
                    
                    # Process images with progress bar
                    image_count_before = len(images)
                    for img_path in tqdm(image_files, desc=f"Processing images from {file_name}"):
                        # Stop if we've reached the maximum number of images
                        if max_images and len(images) >= max_images:
                            break
                            
                        try:
                            with open(img_path, 'rb') as img_file:
                                image_data = img_file.read()
                            # Verify image can be opened
                            Image.open(io.BytesIO(image_data))
                            images.append({
                                'image': {'bytes': image_data},
                                'file_name': os.path.basename(img_path)
                            })
                        except Exception as e:
                            logging.warning(f"Skipping corrupted image {os.path.basename(img_path)}: {str(e)}")
                    
                    logging.info(f"Added {len(images) - image_count_before} images from {file_name}")
                    
                except FileNotFoundError:
                    logging.error("7z command not found. Please install 7-Zip with sudo apt-get install p7zip-full.")
                    return
                
                # Mark archive as processed
                update_processed_files(directory, file_name)
                
                # Clean up downloaded files but keep extracted files
                for file in local_files:
                    if os.path.exists(file):
                        os.remove(file)
                
                logging.info(f"Keeping extracted files in {archive_extract_dir}")
                
            except Exception as e:
                logging.error(f"Error processing archive {base_name}: {str(e)}")
                continue
    
    except Exception as e:
        logging.error(f"Error processing directory {directory}: {str(e)}")
    finally:
        # Clean up temporary download directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logging.info("Cleaned up temporary download directory")
    
    return images

def dataset_exists_on_hf(dataset_name: str, token: str) -> bool:
    """Check if a dataset exists on Hugging Face."""
    try:
        load_dataset(dataset_name, token=token)
        return True
    except Exception:
        return False

def upload_to_huggingface(dataset: Dataset, dataset_name: str, token: str):
    """Upload a dataset to Hugging Face."""
    try:
        dataset.push_to_hub(
            dataset_name,
            token=token,
            private=True
        )
    except Exception as e:
        logging.error(f"Error uploading to Hugging Face: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process and reupload split zip archives as separate datasets')
    parser.add_argument('source_repo', type=str, help='Source repository (e.g., "jzousz/GenImage")')
    parser.add_argument('target_org', type=str, help='Target organization (e.g., "bitmind")')
    parser.add_argument('--directories', nargs='+', help='Specific directories to process (optional)')
    parser.add_argument('--hf-token', type=str, required=True, help='Hugging Face token')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--sample-archives', type=int, default=None, help='Process only N archives from each directory')
    
    args = parser.parse_args()
    logger = setup_logger()
    
    try:
        # List all directories in the source repo if not specified
        if not args.directories:
            all_files = list_hf_files(args.source_repo, args.hf_token)
            directories = set()
            for file in all_files:
                if '/' in file:  # File is in a subdirectory
                    directories.add(file.split('/')[0])
            directories = sorted(list(directories))
        else:
            directories = args.directories
        
        logger.info(f"Found directories: {directories}")
        
        # Process each directory
        for directory in directories:
            logger.info(f"\nProcessing directory: {directory}")
            
            # Create dataset name
            source_repo_name = args.source_repo.split('/')[-1]
            dataset_name = f"{args.target_org}/{source_repo_name}_{directory}"
            
            # Check if dataset already exists
            if dataset_exists_on_hf(dataset_name, args.hf_token):
                logger.warning(f"Dataset {dataset_name} already exists. Skipping...")
                continue
            
            # Process directory and get images
            images = process_directory(
                args.source_repo, 
                directory, 
                args.target_org, 
                args.hf_token,
                max_images=args.max_images,
                sample_archives=args.sample_archives
            )
            
            if not images:
                logger.warning(f"No valid images found in {directory}. Skipping...")
                continue
            
            # Create and upload dataset
            logger.info(f"Creating dataset with {len(images)} images...")
            dataset = Dataset.from_list(images)
            
            logger.info(f"Uploading dataset as {dataset_name}...")
            upload_to_huggingface(dataset, dataset_name, args.hf_token)
            logger.info(f"Successfully uploaded {dataset_name}")
            
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()