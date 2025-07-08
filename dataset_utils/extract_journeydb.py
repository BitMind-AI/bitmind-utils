import os
import tarfile
from huggingface_hub import HfApi, hf_hub_download
import shutil
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import sys
from datasets import Dataset, DatasetDict, Image, Features, Value
from PIL import Image as PILImage
import glob
import time
from requests.exceptions import HTTPError
from huggingface_hub.utils import HfHubHTTPError

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging to file and stdout
log_file = os.path.join('logs', 'transfer.log')

def log_progress(message):
    """Helper function to log progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    # Write to both stdout and file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_message)
    sys.stdout.write(formatted_message)
    sys.stdout.flush()
    
def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """
    Retry a function with exponential backoff
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for retry in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (HTTPError, HfHubHTTPError) as e:
                if retry == max_retries - 1:  # Last retry
                    raise
                if "429" in str(e):  # Too Many Requests
                    log_progress(f"Rate limited. Waiting {delay} seconds before retry {retry + 1}/{max_retries}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise
    return wrapper

@retry_with_backoff
def create_repo_with_retry(api, repo_id, repo_type, exist_ok=True):
    return api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=exist_ok)

@retry_with_backoff
def upload_file_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type):
    return api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type
    )

@retry_with_backoff
def push_to_hub_with_retry(dataset, repo_id, split, private, token):
    return dataset.push_to_hub(
        repo_id,
        split=split,
        private=private,
        token=token
    )
    
def process_images_to_dataset(image_dir):
    """
    Convert a directory of images to a Hugging Face dataset
    """
    image_files = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)
    
    # Create dataset dictionary
    data_dict = {
        "image": image_files,
        "file_name": [os.path.basename(f) for f in image_files]
    }
    
    # Create dataset with Image feature
    features = Features({
        "image": Image(),
        "file_name": Value("string")
    })
    
    dataset = Dataset.from_dict(data_dict, features=features)
    return dataset

def download_and_extract(repo_id, path, local_dir):
    """
    Download and extract .tgz files from Hugging Face dataset
    """
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        log_progress(f"Downloading {path}")
        file_path = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
        
        log_progress(f"Extracting {path} to {local_dir}")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=local_dir)
            
        os.remove(file_path)
        return True, local_dir
    except Exception as e:
        log_progress(f"Error processing {path}: {str(e)}")
        return False, local_dir

def upload_split(api, base_dir, new_repo_name, split_name):
    """
    Convert and upload split as a proper Hugging Face dataset
    """
    try:
        log_progress(f"Converting {split_name} split to dataset format")
        
        # Convert images to dataset
        dataset = process_images_to_dataset(base_dir)
        
        # Push to hub
        log_progress(f"Pushing {split_name} split to hub")
        push_to_hub_with_retry(
            new_repo_name,
            split=split_name,
            private=False,
            token=api.token
        )
        
        log_progress(f"Successfully uploaded {split_name} split")
        return True
    except Exception as e:
        log_progress(f"Error uploading {split_name} split: {str(e)}")
        return False

def flatten_directory(base_dir):
    """
    Move all files from subdirectories to the base directory and remove subdirectories
    """
    # Get all files from subdirectories
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        if root != base_dir:  # Skip the base directory itself
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(base_dir, file)
                # Move file to base directory
                shutil.move(src_path, dst_path)
                all_files.append(dst_path)
    
    # Remove all subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    return all_files

class ProgressTracker:
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
        self.progress = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                progress = json.load(f)
                log_progress(f"Loaded previous progress: {progress}")
                return progress
        return {'train': [], 'valid': [], 'test': False}

    def save_progress(self, source_dir, index):
        with self.lock:
            if source_dir == 'test':
                self.progress['test'] = True
            else:
                if index not in self.progress[source_dir]:
                    self.progress[source_dir].append(index)
            with open(self.filename, 'w') as f:
                json.dump(self.progress, f)

    def is_completed(self, source_dir, index):
        with self.lock:
            if source_dir == 'test':
                return self.progress['test']
            return index in self.progress[source_dir]

def process_split(repo_id, source_dir, new_repo_name, progress_tracker, max_workers=8):
    """
    Process an entire split directory
    """
    api = HfApi()
    split_mapping = {
        'train': 'train',
        'valid': 'validation',
        'test': 'test'
    }
    
    base_dir = f"temp_{source_dir}"
    os.makedirs(base_dir, exist_ok=True)
    
    log_progress(f"Starting processing of {source_dir} split")
    
    try:
        # Handle train and valid directories with numbered .tgz files
        if source_dir in ['train', 'valid']:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(200):
                    if not progress_tracker.is_completed(source_dir, i):
                        tgz_path = f"data/{source_dir}/imgs/{i:03d}.tgz"
                        temp_dir = os.path.join(base_dir, f"{i:03d}")
                        futures.append(executor.submit(download_and_extract, repo_id, tgz_path, temp_dir))
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        log_progress(f"Error in batch processing: {str(e)}")
            
            # Flatten the directory structure after all downloads are complete
            log_progress(f"Flattening directory structure for {source_dir}")
            flatten_directory(base_dir)
        
        # Handle test directory with single imgs.tgz
        elif source_dir == 'test' and not progress_tracker.progress['test']:
            download_and_extract(repo_id, "data/test/imgs.tgz", base_dir)
            flatten_directory(base_dir)
        
        # Create the final directory structure
        final_dir = os.path.join("final", split_mapping[source_dir])
        os.makedirs(final_dir, exist_ok=True)
        
        # Move all files to the final directory
        for file in os.listdir(base_dir):
            src_path = os.path.join(base_dir, file)
            if os.path.isfile(src_path):
                dst_path = os.path.join(final_dir, file)
                shutil.move(src_path, dst_path)
        
        # Upload the split as a dataset
        if upload_split(api, final_dir, new_repo_name, split_mapping[source_dir]):
            if source_dir == 'test':
                progress_tracker.save_progress(source_dir, True)
            else:
                for i in range(200):
                    progress_tracker.save_progress(source_dir, i)
        
    finally:
        # Clean up
        shutil.rmtree(base_dir, ignore_errors=True)
        if os.path.exists("final"):
            shutil.rmtree("final")

def create_dataset_metadata(repo_name):
    """
    Create and upload dataset metadata
    """
    log_progress("Creating dataset metadata")
    api = HfApi()
    
    # Create dataset card
    dataset_card = """---
annotations_creators:
- no-annotation
language_creators:
- machine-generated
languages:
- en
licenses:
- unknown
multilinguality:
- monolingual
size_categories:
- n>1M
source_datasets:
- original
task_categories:
- image-generation
task_ids:
- text-to-image
---

# Dataset Card for JourneyDB

This is a mirror of the JourneyDB dataset.
"""
    
    # Upload dataset card
    with open("README.md", "w") as f:
        f.write(dataset_card)
    upload_file_with_retry(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset"
    )
    os.remove("README.md")

def main():
    # Configuration
    source_repo_id = "JourneyDB/JourneyDB"
    new_repo_name = "bitmind/JourneyDB"
    progress_file = "transfer_progress.json"
    max_workers = 8
    
    log_progress("Starting JourneyDB transfer script")
    
    # Create the repository if it doesn't exist
    api = HfApi()
    create_repo_with_retry(api, new_repo_name, "dataset", exist_ok=True)
    
    # Create final directory structure
    os.makedirs("final", exist_ok=True)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(progress_file)
    
    # Process each split
    for dir_name in ['test', 'valid', 'train']:
        process_split(source_repo_id, dir_name, new_repo_name, progress_tracker, max_workers)
    
    # Create dataset metadata
    create_dataset_metadata(new_repo_name)
    log_progress("Script execution completed")

if __name__ == "__main__":
    main()