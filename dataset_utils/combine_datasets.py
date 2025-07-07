import argparse
import re
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import list_datasets, HfApi, hf_hub_download
from utils.hugging_face_utils import dataset_exists_on_hf, upload_to_huggingface
import shutil
import tempfile
import os
import time
import requests


# Add dataset alias mapping to match run_dataset_gen.sh
DATASET_ALIASES = {
    "bm-real": "bm",
    "celeb-a-hq": "celeb",
    "ffhq-256": "ffhq",
    "MS-COCO-unique-256": "coco",
    "AFHQ": "afhq",
    "lfw": "lfw",
    "caltech-256": "c256",
    "caltech-101": "c101",
    "dtd": "dtd",
    "idoc-mugshots-images": "mug"
}


def find_dataset_chunks(hf_org, dataset_name, model_name, hf_token):
    """Find all dataset chunks matching the pattern on Hugging Face."""
    print(f"Searching for dataset chunks for {dataset_name}...")

    # Use the alias mapping directly
    DATASET_ALIASES = {
        "bm-real": "bm",
        "celeb-a-hq": "celeb",
        "ffhq-256": "ffhq",
        "MS-COCO-unique-256": "coco",
        "AFHQ": "afhq",
        "lfw": "lfw",
        "caltech-256": "c256",
        "caltech-101": "c101",
        "dtd": "dtd",
        "idoc-mugshots-images": "mug"
    }
    dataset_alias = DATASET_ALIASES.get(dataset_name, dataset_name)
    print(f"Using dataset alias: {dataset_alias}")
    print(f"Model name: {model_name}")

    # Only match datasets with the correct alias and model name prefix
    pattern = re.compile(rf"{hf_org}/{re.escape(dataset_alias)}_{re.escape(model_name)}_(\\d+)to(\\d+)$")
    all_datasets = list_datasets(author=hf_org, token=hf_token)
    dataset_ids = [ds.id for ds in all_datasets]
    matching_datasets = []
    print(f"Regex pattern: {pattern.pattern}")
    for ds_id in dataset_ids:
        ds_id_stripped = ds_id.strip()
        m = pattern.match(ds_id_stripped)
        if m:
            start_idx = int(m.group(1))
            end_idx = int(m.group(2))
            matching_datasets.append((start_idx, end_idx, ds_id))
        else:
            # Fallback: substring check
            expected_prefix = f"{hf_org}/{dataset_alias}_{model_name}_"
            if ds_id_stripped.startswith(expected_prefix):
                m2 = re.search(r"_(\d+)to(\d+)$", ds_id_stripped)
                if m2:
                    start_idx = int(m2.group(1))
                    end_idx = int(m2.group(2))
                    matching_datasets.append((start_idx, end_idx, ds_id))
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
    parser.add_argument('--media_type', type=str, default='video', choices=['image', 'video'],
                        help='Type of media to combine and upload (image or video).')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    dataset_alias = DATASET_ALIASES.get(args.dataset_name, args.dataset_name)
    # Find all dataset chunks on Hugging Face
    dataset_chunks = find_dataset_chunks(args.hf_org, args.dataset_name, args.model_name, args.hf_token)
    if not dataset_chunks:
        print("No dataset chunks found. Exiting.")
        return
    # Use the min/max indices from the found chunks for naming
    start_index = dataset_chunks[0][0]
    end_index = dataset_chunks[-1][1]
    combined_dataset_name = f"{args.hf_org}/{dataset_alias}_{args.model_name}_{start_index}to{end_index}"
    if args.px:
        combined_dataset_name += f"_{args.px}"
    media_type = args.media_type
    if media_type == 'video':
        valid_extensions = ('.mp4',)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                api = HfApi()
                for start_idx, end_idx, dataset_id in dataset_chunks:
                    print(f"Listing files in repo: {dataset_id}")
                    files = api.list_repo_files(dataset_id, repo_type="dataset", token=args.hf_token)
                    mp4_files = [f for f in files if f.lower().endswith('.mp4')]
                    for mp4_file in mp4_files:
                        print(f"Downloading {mp4_file} from {dataset_id}")
                        local_path = hf_hub_download(dataset_id, mp4_file, repo_type="dataset", token=args.hf_token, local_dir=temp_dir)
                print(f"All {media_type} files downloaded to {temp_dir}. Uploading to Hugging Face...")
                api = HfApi()
                for fname in sorted(os.listdir(temp_dir)):
                    if fname.lower().endswith('.mp4'):
                        local_path = os.path.join(temp_dir, fname)
                        print(f"Uploading {local_path} to {combined_dataset_name} ...")
                        max_retries = 5
                        for attempt in range(max_retries):
                            try:
                                api.upload_file(
                                    path_or_fileobj=local_path,
                                    path_in_repo=fname,
                                    repo_id=combined_dataset_name,
                                    repo_type="dataset",
                                    token=args.hf_token
                                )
                                break  # Success
                            except requests.exceptions.HTTPError as e:
                                if e.response is not None and e.response.status_code == 429:
                                    wait_time = 60
                                    print(f"Rate limited (429). Waiting {wait_time} seconds before retrying (attempt {attempt+1}/{max_retries})...")
                                    time.sleep(wait_time)
                                else:
                                    print(f"Upload failed for {fname}: {e}")
                                    break
                            except Exception as e:
                                print(f"Upload failed for {fname}: {e}")
                                break
                        else:
                            print(f"Failed to upload {fname} after {max_retries} attempts.")
                        time.sleep(1)  # Add a 1-second delay between uploads
                print(f"Finished uploading {combined_dataset_name} to Hugging Face.")
        except Exception as e:
            print(f"Error combining datasets: {str(e)}")
    else:
        # IMAGE: Ccombine as datasets, then push
        try:
            dataset_parts = []
            for start_idx, end_idx, dataset_id in dataset_chunks:
                print(f"Loading dataset: {dataset_id}")
                ds = load_dataset(dataset_id)
                dataset_parts.append(ds['train'])
                print(f"Loaded dataset: {dataset_id}")
            if not dataset_parts:
                raise ValueError("No datasets were successfully loaded. Cannot combine empty list.")
            from datasets import concatenate_datasets
            combined_dataset = concatenate_datasets(dataset_parts, split='train')
            print(f"All datasets combined successfully. Total size: {len(combined_dataset)} items.")
            # Check if combined dataset already exists
            from utils.hugging_face_utils import dataset_exists_on_hf, upload_to_huggingface
            if dataset_exists_on_hf(combined_dataset_name, args.hf_token):
                print(f"Combined dataset {combined_dataset_name} already exists on Hugging Face.")
                user_input = input("Do you want to overwrite it? (y/n): ")
                if user_input.lower() != 'y':
                    print("Operation cancelled.")
                    return
            print(f"Uploading {combined_dataset_name} to Hugging Face...")
            upload_to_huggingface(combined_dataset, combined_dataset_name, args.hf_token)
            print(f"Finished uploading {combined_dataset_name} to Hugging Face.")
        except Exception as e:
            print(f"Error combining datasets: {str(e)}")


if __name__ == "__main__":
    main()