import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL.Image
from datasets import load_dataset, DatasetDict
from datasets.utils.file_utils import get_datasets_user_agent
from huggingface_hub import HfApi, Repository
import os
import json

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=None, retries=0, output_dir=None, index=None):
    """Downloads an image and saves it locally."""
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                str(image_url),
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                if isinstance(image, PIL.Image.Image):
                    # Save image locally
                    image_path = os.path.join(output_dir, f"{index}.jpg")
                    image.save(image_path)
                    return image_path  # Return the local path of the saved image
                else:
                    print(f"Invalid image object for URL: {image_url}")
                    return None
        except Exception as e:
            #print(f"Error fetching image {image_url}: {e}")
            return None

def fetch_images(batch, num_threads, timeout=None, retries=0, output_dir=None):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries, output_dir=output_dir)
    
    # We need to pass a single tuple to the partial function, containing the image URL and the index
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(lambda x: fetch_single_image_with_args(image_url=x[0], index=x[1]), zip(batch['image_url'], batch['index'])))
    return results

def process_and_save_images(dataset, chunk_size, num_threads, timeout, retries, output_dir):
    """Process the dataset in chunks, downloading images in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    current_index = 0
    batch = {
        "image_url": [],
        "index": [],
    }
    
    for i, example in enumerate(dataset):
        batch["image_url"].append(example["url"])
        batch["index"].append(current_index)
        current_index += 1

        # When the batch size reaches the chunk_size, process and save
        if len(batch["image_url"]) == chunk_size:
            print(f"Processing batch {i // chunk_size + 1}")
            fetch_images(batch, num_threads, timeout, retries, output_dir)
            yield batch
            batch = {
                "image_url": [],
                "index": [],
            }

    # Process any remaining images in the final batch
    if len(batch["image_url"]) > 0:
        print(f"Processing final batch")
        fetch_images(batch, num_threads, timeout, retries, output_dir)
        yield batch

def load_and_push_to_hub(output_dir, repo_id, token, chunk_idx):
    """Loads the dataset from the local directory and pushes it to the Hugging Face Hub."""
    api = HfApi()
    
    # Create or access the destination dataset repository
    repo_url = api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    # Clone the dataset repository to a local directory
    repo = Repository(local_dir=f"./{repo_id}", clone_from=repo_url, repo_type="dataset", token=token)
    
    # Load the image folder as a dataset
    image_dataset = load_dataset("imagefolder", data_dir=output_dir)
    
    # Add the 'original_index' column by extracting the index from the filename
    def extract_index_from_filename(example):
        # example['image'] is an image object, but it also has a 'path' attribute that contains the file path
        
        filename = example['image'].filename  # Access the filename from the image object
        original_index = int(os.path.splitext(os.path.basename(filename))[0])
        example["original_index"] = original_index
        return example

    # Apply the function to extract the original index from the filenames
    image_dataset = image_dataset.map(extract_index_from_filename)['train']

    # Push the dataset to the Hub with a new split for each chunk (e.g., train_0, train_1, ...)
    config_name = f"chunk_{chunk_idx}"
    # Retry logic for pushing the dataset
    retry_attempts = 5
    for attempt in range(retry_attempts):
        try:
            image_dataset.push_to_hub(repo_id, config_name=config_name, token=token)
            print("Dataset pushed successfully.")
            break  # Exit loop if successful
        except Exception as e:
            if "429" in str(e):  # Check for rate limit error
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("An error occurred:", e)
                break  # Exit on non-rate limit errors

def main(args):
    print(f"Loading dataset from {args.dataset_id} (streaming mode enabled)...")
    dset = load_dataset(args.dataset_id, streaming=True)
    
    output_dir = "./images"
    
    print(f"Processing images in chunks of {args.chunk_size}, with {args.num_threads} threads, retries {args.retries}, timeout {args.timeout} seconds.")
    
    # Process the dataset and save images locally in chunks
    for chunk_idx, batch in enumerate(process_and_save_images(
        dataset=dset["train"],
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        timeout=args.timeout,
        retries=args.retries,
        output_dir=output_dir
    )):
        print(f"Uploading chunk {chunk_idx} from {output_dir} to Hugging Face Hub...")
        
        # Load the dataset as an image folder and push to the hub for each chunk
        load_and_push_to_hub(output_dir, args.dest_repo_id, args.token, chunk_idx)
        
        # Optionally, delete images after uploading to save space
        # for file in os.listdir(output_dir):
        #     os.remove(os.path.join(output_dir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and upload images from a dataset.")
    
    parser.add_argument("--dataset_id", type=str, default="bitmind/open-images-v7", help="The dataset ID to load (default: 'bitmind/open-images-v7').")
    parser.add_argument("--num_threads", type=int, default=os.cpu_count() - 1, help="Number of threads for processing images")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries for downloading each image (default: 0).")
    parser.add_argument("--timeout", type=int, default=1, help="Timeout in seconds for image requests (default: None).")
    parser.add_argument("--chunk_size", type=int, default=15000, help="Number of rows to process before uploading (default: 1000).")
    parser.add_argument("--dest_repo_id", type=str, default="bitmind/open-images-v7-jpg", help="Destination Hugging Face dataset repository ID (default: 'bitmind/open-images-v7-jpg').")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token for authentication and pushing to the repository.")
    
    args = parser.parse_args()
    main(args)