import argparse
import fiftyone as fo
import fiftyone.zoo as foz
from huggingface_hub import HfApi
from datasets import load_dataset, DatasetDict


def download_and_upload_split(hf_token, hf_repo_id, split):
    """
    Downloads a split of the Open Images V7 dataset and uploads it to a specified Hugging Face Dataset repository.

    Parameters:
        hf_token (str): Hugging Face API token.
        hf_repo_id (str): Hugging Face repository ID (e.g., "username/repo_name").
        split (str): Dataset split to download and upload (e.g., "train", "validation", "test").
    """
    print(f"Loading dataset split {split}")
    # Load the dataset split from FiftyOne Zoo
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        dataset_name=f"open-images-v7-{split}",
        label_types=[],  # Only load images, no labels
    )
    local_dir = f"/../../fiftyone/open-images-v7/{split}/data/"
    
    print(f"Loading image dataset split {split}")
    image_dataset = load_dataset("imagefolder", data_dir=local_dir, split=split)
    print(f"Pushing image dataset split {split} to {hf_repo_id}")
    image_dataset.push_to_hub(hf_repo_id, token=token)
    print(f"Uploaded {split} split to {hf_repo_id} successfully.")

def main():
    """
    Main function to parse arguments and upload each dataset split.
    """
    parser = argparse.ArgumentParser(description="Process and upload images from Open Images V7 subset.")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--repo_id", type=str, default='bitmind/open-images-v7-subset', help="Hugging Face repository ID (e.g., 'username/repo_name')")

    args = parser.parse_args()

    # Define the splits to download and upload
    splits = ["validation", "test", "train"]

    # Loop through each split and upload it
    for split in splits:
        download_and_upload_split(args.token, args.repo_id, split)

    print("All splits uploaded successfully.")


if __name__ == "__main__":
    main()