import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

def reorganize_and_zip(source_dir: Path, output_dir: Path):
    """
    Create a zip archive for each sav_XXX directory.
    
    Args:
        source_dir: Directory containing extracted sav_XXX folders
        output_dir: Directory to save the reorganized zip files
    """
    output_dir.mkdir(exist_ok=True)
    
    # Process each sav_XXX directory
    for sav_dir in sorted(source_dir.glob("sav_*")):
        if not sav_dir.is_dir():
            continue
            
        print(f"\nProcessing {sav_dir.name}")
        
        # Collect all video and json files for this directory
        all_files = []
        video_dir = sav_dir / "sav_train" / sav_dir.name
        if not video_dir.exists():
            print(f"Warning: Expected directory not found: {video_dir}")
            continue
            
        for video_file in video_dir.glob("*.mp4"):
            video_name = video_file.stem
            auto_json = video_file.with_name(f"{video_name}_auto.json")
            manual_json = video_file.with_name(f"{video_name}_manual.json")
            
            if video_file.exists():
                all_files.append({
                    'video': video_file,
                    'auto_json': auto_json if auto_json.exists() else None,
                    'manual_json': manual_json if manual_json.exists() else None
                })

        if not all_files:
            print(f"Warning: No files found in {video_dir}")
            continue

        # Create zip file for this directory
        zip_path = output_dir / f"{sav_dir.name}.zip"
        print(f"Creating {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_group in tqdm(all_files, desc=f"Adding files to {sav_dir.name}.zip"):
                # Store video with a flattened path
                video_name = file_group['video'].name
                zf.write(file_group['video'], video_name)
                
                # Store metadata in a metadata subdirectory
                if file_group['auto_json']:
                    zf.write(file_group['auto_json'], f"metadata/{video_name}_auto.json")
                if file_group['manual_json']:
                    zf.write(file_group['manual_json'], f"metadata/{video_name}_manual.json")
        
        print(f"Created {zip_path} with {len(all_files)} videos")

if __name__ == "__main__":
    source_dir = Path("extracted")
    output_dir = Path("upload_ready")
    reorganize_and_zip(source_dir, output_dir)