import os
import random
from datasets import load_dataset
from PIL import Image
import requests
import itertools
import argparse
import time
from datetime import datetime

# Add argument parser
parser = argparse.ArgumentParser(description='Download random images from HuggingFace dataset')
parser.add_argument('--token', type=str, required=True, help='HuggingFace API token')
args = parser.parse_args()

# Parameters
dataset_name = "bitmind/open-images-v7-subset-256-flat"
split_name = "train"
output_dir = "random_images"
num_images = 1000

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get dataset info first
dataset = load_dataset(dataset_name, split=split_name, streaming=True, token=args.token)
total_size = dataset.info.splits[split_name].num_examples

# Function to download and save an image
def save_image(data, output_path):
    try:
        save_start = time.time()
        image = data["image"]
        image.save(output_path)
        save_time = time.time() - save_start
        print(f"Saved: {output_path} (save_time: {save_time:.2f}s)")
        return save_time
    except Exception as e:
        print(f"Failed to save image at {output_path}: {e}")
        return 0

# Track overall metrics
total_start = time.time()
metrics = {
    'shuffle_times': [],
    'sample_times': [],
    'save_times': []
}

# Sample images one at a time with reshuffling
for i in range(num_images):
    try:
        iter_start = time.time()
        
        # Generate new random seed for each image
        seed = random.randint(0, total_size)
        print(f"\nImage {i+1}/{num_images} - Using random seed: {seed}")
        
        # Time the dataset shuffling
        shuffle_start = time.time()
        shuffled_dataset = load_dataset(
            dataset_name, 
            split=split_name, 
            streaming=True, 
            token=args.token
        ).shuffle(seed=seed)
        shuffle_time = time.time() - shuffle_start
        metrics['shuffle_times'].append(shuffle_time)
        
        # Time the sample extraction
        sample_start = time.time()
        sample = next(iter(shuffled_dataset))
        sample_time = time.time() - sample_start
        metrics['sample_times'].append(sample_time)
        
        # Time the save operation
        output_path = os.path.join(output_dir, f"random_image_{i + 1}.jpg")
        save_time = save_image(sample, output_path)
        metrics['save_times'].append(save_time)
        
        # Print metrics for this iteration
        iter_time = time.time() - iter_start
        print(f"Metrics for image {i+1}:")
        print(f"  Shuffle time: {shuffle_time:.2f}s")
        print(f"  Sample time: {sample_time:.2f}s")
        print(f"  Save time: {save_time:.2f}s")
        print(f"  Total iteration time: {iter_time:.2f}s")
        
    except Exception as e:
        print(f"Failed to fetch or save image: {e}")

# Print overall metrics
total_time = time.time() - total_start
print("\nOverall metrics:")
print(f"Total time: {total_time:.2f}s")
print(f"Average time per image: {total_time/num_images:.2f}s")
print(f"Average shuffle time: {sum(metrics['shuffle_times'])/len(metrics['shuffle_times']):.2f}s")
print(f"Average sample time: {sum(metrics['sample_times'])/len(metrics['sample_times']):.2f}s")
print(f"Average save time: {sum(metrics['save_times'])/len(metrics['save_times']):.2f}s")
