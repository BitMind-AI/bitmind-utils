import os
import tarfile
import concurrent.futures
from tqdm import tqdm

def extract_tar(filename):
    """Extract a tar file with progress tracking"""
    try:
        # Create a directory named after the tar file (without .tar extension)
        extract_dir = os.path.join("extracted", os.path.splitext(os.path.basename(filename))[0])
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract to the specific directory
        with tarfile.open(filename) as tar:
            tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {str(e)}")
        return False

def main():
    # Create extracted directory if it doesn't exist
    os.makedirs("extracted", exist_ok=True)

    # Get list of downloaded tar files
    downloads_dir = "downloads"
    tar_files = [f for f in os.listdir(downloads_dir) if f.endswith('.tar')]
    
    print(f"\nExtracting {len(tar_files)} files...")
    
    # Extract tar files in parallel (3 at a time to avoid overwhelming the system)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        with tqdm(total=len(tar_files), desc="Extracting files") as pbar:
            future_to_tar = {
                executor.submit(
                    extract_tar,
                    os.path.join(downloads_dir, filename)
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