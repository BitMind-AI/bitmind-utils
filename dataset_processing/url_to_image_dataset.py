import requests
from PIL import Image
from io import BytesIO
import time
import os
import argparse
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, Repository

def download_image(url, retries=5, max_wait=5):
    """
    Downloads an image from a URL with exponential backoff retry logic.

    Args:
        url (str): The URL of the image to download.
        retries (int): The number of times to retry downloading if it fails.
        max_wait (int): Maximum cumulative wait time in seconds before skipping the image.

    Returns:
        PIL.Image or None: The downloaded and opened image if successful, None if failed.
    """
    backoff_time = 1
    total_wait_time = 0
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                raise Exception(f"Failed to download image: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            time.sleep(backoff_time)
            total_wait_time += backoff_time
            backoff_time *= 2
            if total_wait_time >= max_wait:
                print(f"Skipping {url} after {total_wait_time} seconds of retries.")
                return None
    return None


def process_image(url, max_wait):
    """
    Downloads and resizes an image from a URL to 256x256.

    Args:
        url (str): The URL of the image to download.
        max_wait (int): Maximum cumulative wait time in seconds before skipping the image.

    Returns:
        PIL.Image or None: The resized image if successful, None if failed.
    """
    image = download_image(url, max_wait=max_wait)
    if image:
        try:
            # Convert to RGB and resize to 256x256
            image = image.convert('RGB').resize((256, 256))
            return image
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    return None


def upload_batch(processed_data, destination_dataset_name, hf_token):
    """
    Uploads a batch of processed images to a new Hugging Face dataset.

    Args:
        processed_data (dict): Dictionary containing image metadata and file paths.
        destination_dataset_name (str): Name of the new Hugging Face dataset to upload to.
        hf_token (str): Hugging Face authentication token.
    """
    # Check if the dataset exists, if not create it
    try:
        existing_dataset = load_dataset(destination_dataset_name, split='train')
        new_data = Dataset.from_dict(processed_data)

        # Concatenate the new batch with the existing dataset
        updated_dataset = concatenate_datasets([existing_dataset, new_data])
    except FileNotFoundError:
        # If the dataset doesn't exist yet, use the new data as the dataset
        print(f"Dataset {destination_dataset_name} not found, creating a new one.")
        updated_dataset = Dataset.from_dict(processed_data)

    # Push the updated dataset to the new destination dataset
    updated_dataset.push_to_hub(destination_dataset_name, token=hf_token)

def main(args):
    """
    Main function to process image URLs from a dataset, resize images,
    and upload them to Hugging Face in batches.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    HF_TOKEN = args.hf_token
    BATCH_SIZE = args.batch_size
    MAX_BACKOFF_TIME = args.max_backoff_time
    URL_DATASET_NAME = args.url_dataset_name
    DESTINATION_DATASET_NAME = args.destination_dataset_name

    # Load the original dataset from Hugging Face
    dataset = load_dataset(URL_DATASET_NAME)

    # Directory to store images temporarily
    output_dir = "processed_images"
    os.makedirs(output_dir, exist_ok=True)

    processed_data = {
        "original_index": [],
        "url": [],
        "image_file": []
    }

    for i, item in enumerate(tqdm(dataset)):
        url = item['url']  # Adjust if 'url' is under a different field

        # The original index should be the current index of the image in the dataset
        original_index = i

        # Process the image
        image = process_image(url, max_wait=MAX_BACKOFF_TIME)

        if image:
            image_filename = f"{i}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path, format="JPEG")

            processed_data["original_index"].append(original_index)
            processed_data["url"].append(url)
            processed_data["image_file"].append(image_path)

        # Upload every BATCH_SIZE images
        if (i + 1) % BATCH_SIZE == 0:
            upload_batch(processed_data, DESTINATION_DATASET_NAME, HF_TOKEN)

            # Clear processed data to start the next batch
            processed_data = {
                "original_index": [],
                "url": [],
                "image_file": []
            }

    # Final upload for any remaining images
    if len(processed_data["original_index"]) > 0:
        upload_batch(processed_data, DESTINATION_DATASET_NAME, HF_TOKEN)


if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(
        description="Process image URLs from a dataset and upload the processed images to a new Hugging Face dataset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Number of images to process before each batch upload."
    )
    parser.add_argument(
        "--max_backoff_time",
        type=int,
        default=5,
        help="Maximum backoff time in seconds before skipping a failed image."
    )
    parser.add_argument(
        "--url_dataset_name",
        type=str,
        required=True,
        help="Name of the Hugging Face dataset (e.g., your-username/your-original-dataset) from which to load the data."
    )
    parser.add_argument(
        "--destination_dataset_name",
        type=str,
        required=True,
        help="Name of the new Hugging Face dataset (e.g., your-username/your-new-dataset) to upload processed images."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Your Hugging Face authentication token."
    )

    args = parser.parse_args()
    main(args)
