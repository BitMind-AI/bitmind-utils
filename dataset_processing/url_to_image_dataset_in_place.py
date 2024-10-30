import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL.Image
from datasets import load_dataset, Dataset
from datasets.utils.file_utils import get_datasets_user_agent
from huggingface_hub import HfApi, Repository
import os

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                str(image_url),
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                print(image)
                if isinstance(image, PIL.Image.Image):
                    print("Image!")
                    return image
                else:
                    print(f"Invalid image object for URL: {image_url}")
                    return None
        except Exception as e:
            return None

def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["url"]))
    return batch

def process_and_upload_in_chunks(dataset, chunk_size, num_threads, timeout, retries, repo_id, token):
    api = HfApi()
    
    # Create or access the dataset repository
    repo_url = api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    # Clone the dataset repository to a local directory
    repo = Repository(local_dir=f"./{repo_id}", clone_from=repo_url, repo_type="dataset", token=token)
    
    # Load the existing dataset if it exists
    try:
        existing_dataset = load_dataset(repo_id, use_auth_token=token)
        print(f"Existing dataset loaded from {repo_id}.")
    except Exception as e:
        existing_dataset = None
        print(f"Loading existing dataset {repo_id} failed: {e}")

    chunk = []
    chunk_index = 0

    dataset = dataset.map(lambda example, idx: {"original_index": idx}, with_indices=True)
    # Iterate over the dataset stream and process in chunks
    for i, batch in enumerate(dataset):
        # Append each batch to the chunk
        chunk.append(batch)

        # Once we hit the chunk size, process and upload the chunk
        if (i + 1) % chunk_size == 0:
            print(f"Processing chunk {chunk_index}...")
            chunk_dset = Dataset.from_dict({
                "url": [item["url"] for item in chunk],
                "original_index": [item["original_index"] for item in chunk],
            })
            # Process images in the chunk
            chunk_dset = chunk_dset.map(
                fetch_images, 
                batched=True, 
                batch_size=100,  # Adjust based on memory constraints
                fn_kwargs={"num_threads": num_threads, "timeout": timeout, "retries": retries}
            )
            
            # Filter out failed downloads (rows where 'image' is None)
            chunk_dset = chunk_dset.filter(lambda x: x["image"] is not None)
            chunk_dset = chunk_dset.remove_columns("url")
            print(f"chunk_dset: {chunk_dset}")

            # Concatenate with existing dataset if it exists
            if existing_dataset:
                print(f"Appending chunk {chunk_index} to existing dataset.")
                combined_dset = Dataset.concatenate_datasets([existing_dataset, chunk_dset])
            else:
                print(f"Starting new dataset with chunk {chunk_index}.")
                combined_dset = chunk_dset

            # Push the combined dataset back to the Hugging Face Hub
            print(f"Uploading combined dataset (including chunk {chunk_index}) to Hugging Face.")
            combined_dset.push_to_hub(repo_id, token=token)

            # Reset chunk
            chunk = []
            chunk_index += 1

    # Process any remaining images that didn't fill up the last chunk
    if chunk:
        print(f"Processing remaining images in final chunk {chunk_index}...")
        chunk_dset = Dataset.from_dict({
            "url": [item["url"] for item in chunk],
            "original_index": [item["original_index"] for item in chunk],
        })
        
        chunk_dset = chunk_dset.map(
            fetch_images, 
            batched=True, 
            batch_size=100,  
            fn_kwargs={"num_threads": num_threads, "timeout": timeout, "retries": retries}
        )
        chunk_dset = chunk_dset.filter(lambda x: x["image"] is not None)
        chunk_dset = chunk_dset.remove_columns("url")
        print(f"chunk_dset: {chunk_dset}")
        # Concatenate with existing dataset if it exists
        if existing_dataset:
            combined_dset = Dataset.concatenate_datasets([existing_dataset, chunk_dset])
        else:
            combined_dset = chunk_dset

        # Push the final combined dataset to the Hugging Face Hub
        print(f"Uploading final combined dataset to Hugging Face.")
        combined_dset.push_to_hub(repo_id, token=token)


def main(args):
    print(f"Loading dataset from {args.dataset_id} (streaming mode enabled)...")
    dset = load_dataset(args.dataset_id, streaming=True)
    print(f"URL dataset: {dset}")

    print(f"Processing images in chunks of {args.chunk_size}, with {args.num_threads} threads, batch size {args.batch_size}, retries {args.retries}, timeout {args.timeout} seconds.")
    
    # Process and upload dataset in chunks
    process_and_upload_in_chunks(
        dataset=dset["train"],  # Assuming you're working with a split named 'train'
        chunk_size=args.chunk_size,
        num_threads=args.num_threads,
        timeout=args.timeout,
        retries=args.retries,
        repo_id=args.dest_repo_id,
        token=args.token
    )
    print("Processing and uploading complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and upload images from a dataset.")
    
    parser.add_argument("--dataset_id", type=str, default="bitmind/open-images-v7", help="The dataset ID to load (default: 'bitmind/open-images-v7').")
    parser.add_argument("--num_threads", type=int, default=os.cpu_count() - 1, help="Number of threads for processing images")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing images (default: 100).")
    parser.add_argument("--retries", type=int, default=0, help="Number of retries for downloading each image (default: 0).")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for image requests (default: None).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of rows to process before uploading (default: 1000).")
    parser.add_argument("--dest_repo_id", type=str, default="bitmind/open-images-v7-jpg", help="Destination Hugging Face dataset repository ID (default: 'bitmind/open-images-v7-jpg').")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token for authentication and pushing to the repository.")
    
    args = parser.parse_args()
    main(args)
