import os
import boto3
from tqdm import tqdm
import argparse
import concurrent.futures

# --- S3 CONFIGURATION ---
DEFAULT_S3_BUCKET = "subnet-34-storage"  # Change as needed
DEFAULT_S3_PREFIX = "segmentation/"    # Change as needed
DEFAULT_AWS_PROFILE = "BitmindS3Access-891377275001"  # Change as needed

NPY_EXTENSION = ".npy"

def is_npy_file(key):
    ext = os.path.splitext(key)[-1].lower()
    return ext == NPY_EXTENSION

def download_file_from_s3(s3_client, bucket, s3_key, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, s3_key, local_path)

def main():
    parser = argparse.ArgumentParser(description="Download all .npy files from an S3 directory, preserving structure.")
    parser.add_argument("local_dir", help="Path to the local destination directory")
    parser.add_argument("--s3-bucket", type=str, default=DEFAULT_S3_BUCKET, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default=DEFAULT_S3_PREFIX, help="S3 prefix (directory) to download from")
    parser.add_argument("--profile", type=str, default=DEFAULT_AWS_PROFILE, help="AWS profile to use")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download workers (default: 8)")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    # List all .npy files under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    files_to_download = []
    for page in paginator.paginate(Bucket=args.s3_bucket, Prefix=args.s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if is_npy_file(key):
                rel_path = os.path.relpath(key, args.s3_prefix)
                local_path = os.path.join(args.local_dir, rel_path)
                files_to_download.append((key, local_path))

    print(f"Found {len(files_to_download)} .npy files to download.")

    def download_wrapper(args_tuple):
        s3_key, local_path = args_tuple
        try:
            download_file_from_s3(s3, args.s3_bucket, s3_key, local_path)
        except Exception as e:
            print(f"Error downloading s3://{args.s3_bucket}/{s3_key} to {local_path}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(download_wrapper, files_to_download), total=len(files_to_download), desc="Downloading .npy files"))

if __name__ == "__main__":
    main() 