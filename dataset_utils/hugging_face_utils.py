import os
import json
from datasets import load_dataset
from huggingface_hub import HfApi
import time
import requests

def dataset_exists_on_hf(hf_dataset_name, token):
    """Check if the dataset exists on Hugging Face."""
    api = HfApi()
    try:
        dataset_info = api.dataset_info(hf_dataset_name, token=token)
        return True
    except Exception as e:
        return False

def numerical_sort(value):
    return int(os.path.splitext(os.path.basename(value))[0])

def load_and_sort_dataset(data_dir, file_type):
    # Get list of filenames in the directory with the given extension
    try:
        if file_type == 'image':
            # List image filenames with common image extensions
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
            filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                         if f.lower().endswith(valid_extensions)]
        elif file_type == 'json':
            # List json filenames
            filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                         if f.lower().endswith('.json')]
        elif file_type == 'video':
            # List video filenames (mp4 only for now)
            valid_extensions = ('.mp4',)
            filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                         if f.lower().endswith(valid_extensions)]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if not filenames:
            raise FileNotFoundError(f"No files with the extension '{file_type}' \
                                    found in directory '{data_dir}'")
    
        # Sort filenames numerically (0, 1, 2, 3, 4). Necessary because
        # HF datasets are ordered by string (0, 1, 10, 11, 12). 
        sorted_filenames = sorted(filenames, key=numerical_sort)
        
        # Load the dataset with sorted filenames
        if file_type == 'image':
            return load_dataset("imagefolder", data_files=sorted_filenames)
        elif file_type == 'json':
            return load_dataset("json", data_files=sorted_filenames)
        elif file_type == 'video':
            return load_dataset("videofolder", data_files=sorted_filenames)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def upload_to_huggingface(dataset, repo_name, token, private=False):
    """Uploads the dataset dictionary to Hugging Face."""
    api = HfApi()
    # Create the repo if it doesn't exist, or continue if it does
    api.create_repo(repo_name, repo_type="dataset", token=token, private=private, exist_ok=True)
    # Push the dataset, always pass the token for private repos
    dataset.push_to_hub(repo_name, token=token)

def slice_dataset(dataset, start_index=0, end_index=None):
    """
    Slice a dataset from start_index to end_index (inclusive).
    
    Args:
        dataset: The dataset to slice
        start_index (int): The starting index (inclusive)
        end_index (int): The ending index (inclusive)
    """
    if end_index is not None:
        return dataset.select(range(start_index, end_index + 1))
    return dataset.select(range(start_index, len(dataset)))

def save_as_json(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    # Iterate through rows in dataframe
    for index, row in df.iterrows():
        file_path = os.path.join(output_dir, f"{row['id']}.json")
        # Convert the row to a dictionary and save it as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(row.to_dict(), f, ensure_ascii=False, indent=4)

def upload_videos_as_files(local_dir, repo_id, token, repo_type="dataset", private=False):
    """
    Upload all .mp4 files in local_dir to the specified Hugging Face repo as individual files.
    """
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, token=token, private=private, exist_ok=True)
    for fname in sorted(os.listdir(local_dir)):
        if fname.lower().endswith(".mp4"):
            local_path = os.path.join(local_dir, fname)
            print(f"Uploading {local_path} to {repo_id} ...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        token=token
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
    print("All videos uploaded as files.")