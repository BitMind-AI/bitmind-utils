import os
import zipfile
import argparse
import concurrent.futures

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif", ".npy"}

def is_image_file(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in IMAGE_EXTENSIONS

def zip_subdir(entry, main_dir):
    zip_path = os.path.join(main_dir, f"{entry.name}.zip")
    print(f"[START] Zipping {entry.name} -> {zip_path}")
    count = 0
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(entry.path):
            for file in files:
                if is_image_file(file):
                    abs_path = os.path.join(root, file)
                    arcname = os.path.relpath(abs_path, entry.path)
                    zipf.write(abs_path, arcname)
                    count += 1
    print(f"[DONE]  {entry.name}: {count} images zipped to {zip_path}")
    return entry.name, count, zip_path

def zip_images_in_subdirs(main_dir, workers):
    subdirs = [entry for entry in os.scandir(main_dir) if entry.is_dir()]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_entry = {executor.submit(zip_subdir, entry, main_dir): entry for entry in subdirs}
        for future in concurrent.futures.as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"[ERROR] {entry.name}: {exc}")
    print("\nSummary:")
    for name, count, zip_path in results:
        print(f"{name}: {count} images -> {zip_path}")

def main():
    parser = argparse.ArgumentParser(description="Create a zip for each subdirectory, containing all image files (preserving structure), using parallel workers.")
    parser.add_argument("main_dir", help="Path to the main directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()
    zip_images_in_subdirs(args.main_dir, args.workers)

if __name__ == "__main__":
    main() 