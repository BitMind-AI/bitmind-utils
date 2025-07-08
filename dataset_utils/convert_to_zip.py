import os
import zipfile
import tarfile
from pathlib import Path

def tar_to_zip(tar_path, zip_path):
    """Convert a tar file to zip format"""
    with tarfile.open(tar_path, 'r') as tar:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f:  # Skip if not a file
                    zip_ref.writestr(member.name, f.read())

def convert_all_tars():
    downloads_dir = Path("downloads")
    output_dir = Path("zips")
    output_dir.mkdir(exist_ok=True)
    
    for tar_file in downloads_dir.glob("*.tar"):
        if tar_file.name.startswith(('sav_', 'videos_')):
            zip_path = output_dir / f"{tar_file.stem}.zip"
            print(f"Converting {tar_file} to {zip_path}")
            tar_to_zip(tar_file, zip_path)

if __name__ == "__main__":
    convert_all_tars()