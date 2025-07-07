import argparse
import os
from huggingface_hub import HfApi, HfFolder

def upload_zip_to_hf(zip_path, repo_id, repo_type="dataset", token=None, path_in_repo=None, private=False):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist or is not a file.")

    if token is None:
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("No Hugging Face token found. Please provide one or run `huggingface-cli login`.")

    if path_in_repo is None:
        path_in_repo = os.path.basename(zip_path)

    api = HfApi(token=token)
    # Create the repo if it doesn't exist, with the correct privacy setting
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    print(f"Uploading {zip_path} to {repo_id} (type: {repo_type}) at {path_in_repo} ...")
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a zip file to a Hugging Face Hub repo.")
    parser.add_argument("zip_path", help="Path to the zip file to upload")
    parser.add_argument("repo_id", help="Hugging Face repo id (e.g. username/repo_name)")
    parser.add_argument("--repo_type", default="dataset", choices=["model", "dataset", "space"], help="Type of repo (default: dataset)")
    parser.add_argument("--token", default=None, help="Hugging Face access token (or set HF_TOKEN env var or run huggingface-cli login)")
    parser.add_argument("--path_in_repo", default=None, help="Path in repo to upload to (default: filename)")
    parser.add_argument("--private", action="store_true", help="Make the repo private if created")
    args = parser.parse_args()

    upload_zip_to_hf(
        zip_path=args.zip_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=args.token,
        path_in_repo=args.path_in_repo,
        private=args.private,
    )