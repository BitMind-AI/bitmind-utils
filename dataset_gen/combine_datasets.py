import argparse
import re
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import list_datasets
from utils.hugging_face_utils import dataset_exists_on_hf, upload_to_huggingface


def find_dataset_chunks(hf_org, dataset_name, model_name, hf_token):
    """Find all dataset chunks matching the pattern on Hugging Face."""
    print(f"Searching for dataset chunks for {dataset_name}...")
    
    # Pattern to match dataset chunks like:
    # bitmind/google-images-holdout-deduped-commits_3___0-to-2346___FLUX.1-dev
    pattern = f"{hf_org}/{dataset_name}___\\d+-to-\\d+___{model_name}"
    
    # List all datasets in the organization
    all_datasets = list_datasets(author=hf_org, token=hf_token)
    dataset_ids = [ds.id for ds in all_datasets]
    
    # Filter datasets matching our pattern
    matching_datasets = []
    for ds_id in dataset_ids:
        if re.match(pattern, ds_id):
            # Extract the indices from the dataset name
            match = re.search(f"{hf_org}/{dataset_name}___(\d+)-to-(\d+)___{model_name}", ds_id)
            if match:
                start_idx = int(match.group(1))
                end_idx = int(match.group(2))
                matching_datasets.append((start_idx, end_idx, ds_id))
    
    # Sort by start index
    matching_datasets.sort(key=lambda x: x[0])
    
    if not matching_datasets:
        print(f"No matching dataset chunks found for {dataset_name} with model {model_name}")
    else:
        print(f"Found {len(matching_datasets)} dataset chunks:")
        for start, end, ds_id in matching_datasets:
            print(f"  {ds_id} ({start} to {end})")
    
    return matching_datasets


def load_and_combine_datasets(dataset_chunks, px=None):
    """Load and combine dataset chunks."""
    dataset_parts = []
    
    # Load each dataset chunk
    for start_idx, end_idx, dataset_id in dataset_chunks:
        try:
            dataset = load_dataset(dataset_id)
            dataset_parts.append(dataset['train'])
            print(f"Loaded dataset: {dataset_id}")
        except Exception as e:
            print(f"Failed to load dataset {dataset_id}: {str(e)}")
    
    if not dataset_parts:
        raise ValueError("No datasets were successfully loaded. Cannot combine empty list.")
    
    # Concatenate all datasets in the correct order
    combined_dataset = concatenate_datasets(dataset_parts, split='train')
    print(f"All datasets combined successfully. Total size: {len(combined_dataset)} items.")
    return combined_dataset


def parse_arguments():
    """
    Before running, authenticate with command line to upload to Hugging Face:
    huggingface-cli login
    
    Do not add token as Git credential.

    Example usage:
    python combine_datasets.py bitmind google-images-holdout-deduped-commits_3 FLUX.1-dev YOUR_HF_TOKEN
    """
    parser = argparse.ArgumentParser(description='Load and combine Hugging Face datasets.')
    parser.add_argument('hf_org', type=str, help='Hugging Face organization name')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('model_name', type=str, help='Name of the diffusion model')
    parser.add_argument('hf_token', type=str, help='Token for uploading to Hugging Face.')
    parser.add_argument('--px', type=int, default=None, help='Dimensions (ex. 256) of images.')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    combined_dataset_name = f"{args.hf_org}/{args.dataset_name}___{args.model_name}"
    if args.px:
        combined_dataset_name += f"___{args.px}"
    
    # Find all dataset chunks on Hugging Face
    dataset_chunks = find_dataset_chunks(args.hf_org, args.dataset_name, args.model_name, args.hf_token)
    
    if not dataset_chunks:
        print("No dataset chunks found. Exiting.")
        return
    
    # Load and combine datasets
    try:
        combined_dataset = load_and_combine_datasets(dataset_chunks, args.px)
        
        # Check if combined dataset already exists
        if dataset_exists_on_hf(combined_dataset_name, args.hf_token):
            print(f"Combined dataset {combined_dataset_name} already exists on Hugging Face.")
            user_input = input("Do you want to overwrite it? (y/n): ")
            if user_input.lower() != 'y':
                print("Operation cancelled.")
                return
        
        # Upload combined dataset
        print(f"Uploading {combined_dataset_name} to Hugging Face...")
        upload_to_huggingface(combined_dataset, combined_dataset_name, args.hf_token)
        print(f"Finished uploading {combined_dataset_name} to Hugging Face.")
        
    except Exception as e:
        print(f"Error combining datasets: {str(e)}")


if __name__ == "__main__":
    main()