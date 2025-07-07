import argparse
import os
from huggingface_hub import HfApi, HfFolder
from upload_zip_to_hf import upload_zip_to_hf

def upload_all_zips_to_hf(local_dir, repo_id, repo_dir=None, repo_type="dataset", token=None, private=False):
    if not os.path.isdir(local_dir):
        raise NotADirectoryError(f"{local_dir} is not a directory.")

    zips = [f for f in os.listdir(local_dir) if f.lower().endswith('.zip')]
    if not zips:
        print(f"No zip files found in {local_dir}.")
        return

    print(f"Found {len(zips)} zip files in {local_dir}.")
    for zip_name in zips:
        zip_path = os.path.join(local_dir, zip_name)
        if repo_dir:
            path_in_repo = os.path.join(repo_dir, zip_name).replace("\\", "/")
        else:
            path_in_repo = zip_name
        print(f"Uploading {zip_path} to {repo_id}:{path_in_repo} ...")
        upload_zip_to_hf(
            zip_path=zip_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            path_in_repo=path_in_repo,
            private=private,
        )
    print("All uploads complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload all zip files in a directory to a directory within a Hugging Face dataset repo.")
    parser.add_argument("local_dir", help="Directory containing zip files to upload")
    parser.add_argument("repo_id", help="Hugging Face repo id (e.g. username/repo_name)")
    parser.add_argument("--repo_dir", default=None, help="Target directory in the repo (default: root)")
    parser.add_argument("--repo_type", default="dataset", choices=["model", "dataset", "space"], help="Type of repo (default: dataset)")
    parser.add_argument("--token", default=None, help="Hugging Face access token (or set HF_TOKEN env var or run huggingface-cli login)")
    parser.add_argument("--private", action="store_true", help="Make the repo private if created")
    args = parser.parse_args()

    upload_all_zips_to_hf(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        repo_dir=args.repo_dir,
        repo_type=args.repo_type,
        token=args.token,
        private=args.private,
    ) 