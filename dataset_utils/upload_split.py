import os
import glob
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset

def upload_json_files_to_split(directory_path, repo_id, split_name, hf_token, private=True):
    """
    Upload all JSON files from a directory and its subdirectories to a Hugging Face repo split.
    Appends to existing split if it exists.
    
    Args:
        directory_path: Path to directory containing JSON files
        repo_id: Hugging Face repository ID (e.g., "org/repo-name")
        split_name: Name of the split to upload to
        hf_token: Hugging Face API token
        private: Whether to make the repository private
    """
    print(f"=== Uploading JSON files to {repo_id}/{split_name} ===")
    
    # Find all JSON files recursively
    json_pattern = os.path.join(directory_path, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {directory_path} or its subdirectories")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Load existing dataset if it exists and has data
    existing_dataset = None
    try:
        existing_dataset = load_dataset(repo_id, token=hf_token)
        print(f"Found existing dataset with splits: {list(existing_dataset.keys())}")
    except Exception as e:
        error_msg = str(e).lower()
        if "doesn't contain any data files" in error_msg:
            print(f"Repository {repo_id} exists but contains no data files - treating as new dataset")
        elif "not found" in error_msg or "does not exist" in error_msg:
            print(f"Repository {repo_id} does not exist - will create new dataset")
        else:
            print(f"Error accessing dataset {repo_id}: {e}")
            print("Will attempt to create/upload as new dataset")
    
    # Load new JSON files
    print(f"Loading {len(json_files)} JSON files...")
    try:
        new_dataset = load_dataset("json", data_files=json_files, split="train")
        print(f"Loaded {len(new_dataset)} records from JSON files")
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        return
    
    # Combine with existing split if it exists and has data
    if existing_dataset and split_name in existing_dataset:
        print(f"Appending to existing {split_name} split...")
        try:
            existing_split = existing_dataset[split_name]
            if len(existing_split) > 0:
                combined_dataset = Dataset.concatenate_datasets([existing_split, new_dataset])
                print(f"Combined dataset: {len(existing_split)} existing + {len(new_dataset)} new = {len(combined_dataset)} total")
            else:
                print(f"Existing {split_name} split is empty - using new data only")
                combined_dataset = new_dataset
        except Exception as e:
            print(f"Error combining datasets: {e}")
            print("Uploading new data only...")
            combined_dataset = new_dataset
    else:
        print(f"Creating new {split_name} split...")
        combined_dataset = new_dataset
    
    # Upload to Hugging Face
    print(f"Uploading {len(combined_dataset)} records to {repo_id}/{split_name}...")
    try:
        combined_dataset.push_to_hub(
            repo_id, 
            split=split_name, 
            token=hf_token, 
            private=private
        )
        print(f"Successfully uploaded to {repo_id}/{split_name}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload JSON files to Hugging Face dataset split")
    parser.add_argument("directory", help="Directory containing JSON files to upload")
    parser.add_argument("repo_id", help="Hugging Face repository ID (e.g., 'org/repo-name')")
    parser.add_argument("split_name", help="Name of the split to upload to")
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument("--public", action="store_true", help="Make repository public (default: private)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    upload_json_files_to_split(
        directory_path=args.directory,
        repo_id=args.repo_id,
        split_name=args.split_name,
        hf_token=args.token,
        private=not args.public
    )

if __name__ == "__main__":
    main()