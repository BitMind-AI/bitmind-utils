import os
import requests
import tarfile
import concurrent.futures
from tqdm import tqdm
import threading

# Create a thread-safe progress bar
class DownloadProgress:
    def __init__(self, total_files):
        self.pbar = tqdm(total=total_files, desc="Downloading files")
        self.lock = threading.Lock()

    def update(self):
        with self.lock:
            self.pbar.update(1)

def download_file(url, filename, progress):
    """Download a single file with progress tracking"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Download with progress
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        progress.update()
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def extract_tar(filename):
    """Extract a tar file with progress tracking"""
    try:
        with tarfile.open(filename) as tar:
            tar.extractall(path="extracted")
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {str(e)}")
        return False

def read_files_from_txt(filename="files.txt"):
    """Read files and URLs from the text file"""
    files = {}
    try:
        with open(filename, 'r') as f:
            # Skip the header line
            next(f)  # Skip 'file_name    cdn_link' line
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    filename, url = parts
                    files[filename] = url
    except Exception as e:
        print(f"Error reading file list: {str(e)}")
        return {}
    return files

def main():
    # Create directories
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("extracted", exist_ok=True)

    # Read files from text file
    files = read_files_from_txt()
    if not files:
        print("No files found to download. Please check files.txt")
        return

    print(f"Starting download of {len(files)} files...")

    # Download files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        progress = DownloadProgress(len(files))
        future_to_file = {
            executor.submit(
                download_file, 
                url, 
                os.path.join("downloads", filename),
                progress
            ): filename 
            for filename, url in files.items()
        }

        # Wait for downloads to complete
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    print(f"Failed to download {filename}")
            except Exception as e:
                print(f"Download failed for {filename}: {str(e)}")

    # Extract tar files
    tar_files = [f for f in os.listdir("downloads") if f.endswith('.tar')]
    print("\nExtracting files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        with tqdm(total=len(tar_files), desc="Extracting files") as pbar:
            future_to_tar = {
                executor.submit(
                    extract_tar,
                    os.path.join("downloads", filename)
                ): filename 
                for filename in tar_files
            }

            for future in concurrent.futures.as_completed(future_to_tar):
                filename = future_to_tar[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"Failed to extract {filename}")
                    pbar.update(1)
                except Exception as e:
                    print(f"Extraction failed for {filename}: {str(e)}")
                    pbar.update(1)

if __name__ == "__main__":
    main()