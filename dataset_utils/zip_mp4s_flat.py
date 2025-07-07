import os
import zipfile
import argparse

def zip_mp4s_flat(input_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.mp4'):
                    abs_path = os.path.join(root, file)
                    # Use only the filename in the zip (flatten)
                    arcname = file
                    # If duplicate filenames, add a number to avoid overwriting
                    if arcname in zipf.namelist():
                        base, ext = os.path.splitext(file)
                        i = 1
                        while f"{base}_{i}{ext}" in zipf.namelist():
                            i += 1
                        arcname = f"{base}_{i}{ext}"
                    zipf.write(abs_path, arcname)
    print(f"Created zip: {output_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zip all mp4 files in a directory (recursively), flattening structure.")
    parser.add_argument("input_dir", help="Directory to search for mp4 files")
    parser.add_argument("output_zip", help="Output zip file path")
    args = parser.parse_args()
    zip_mp4s_flat(args.input_dir, args.output_zip)