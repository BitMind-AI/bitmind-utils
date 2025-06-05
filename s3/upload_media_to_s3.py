import os
import boto3
from tqdm import tqdm
import argparse
import concurrent.futures

# --- S3 CONFIGURATION ---
S3_BUCKET = "subnet-34-storage"  # Change as needed
S3_PREFIX = "semisynthetics/"    # Change as needed, will be prepended to all S3 keys
AWS_PROFILE = "BitmindS3Access-891377275001"  # Change as needed

def upload_file_to_s3(local_path, s3_key, s3_client):
    # Guess content type
    ext = os.path.splitext(local_path)[-1].lower()
    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".wmv": "video/x-ms-wmv",
        ".npy": "application/octet-stream"
    }.get(ext, "application/octet-stream")
    with open(local_path, "rb") as f:
        s3_client.upload_fileobj(f, S3_BUCKET, s3_key, ExtraArgs={"ContentType": content_type})

def main():
    parser = argparse.ArgumentParser(description="Upload all files in a media directory to S3, preserving structure.")
    parser.add_argument("media_dir", help="Path to the root media directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel upload workers (default: 8)")
    parser.add_argument("--s3-prefix", type=str, default=S3_PREFIX, help="S3 prefix to prepend to all keys")
    parser.add_argument("--profile", type=str, default=AWS_PROFILE, help="AWS profile to use")
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")

    # Collect all files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(args.media_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # S3 key: s3_prefix + relative path from media_dir
            rel_path = os.path.relpath(local_path, args.media_dir)
            s3_key = os.path.join(args.s3_prefix, rel_path).replace("\\", "/")
            files_to_upload.append((local_path, s3_key))

    print(f"Found {len(files_to_upload)} files to upload.")

    # Parallel upload with progress bar
    def upload_wrapper(args_tuple):
        local_path, s3_key = args_tuple
        try:
            upload_file_to_s3(local_path, s3_key, s3)
        except Exception as e:
            print(f"Error uploading {local_path} to s3://{S3_BUCKET}/{s3_key}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(upload_wrapper, files_to_upload), total=len(files_to_upload), desc="Uploading media files"))

if __name__ == "__main__":
    main() 