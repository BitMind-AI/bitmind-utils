import os
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

def is_image_file(filename):
    ext = os.path.splitext(filename)[-1].lower()
    return ext in IMAGE_EXTENSIONS

def count_images_in_subdirs(main_dir):
    for entry in os.scandir(main_dir):
        if entry.is_dir():
            count = 0
            for root, _, files in os.walk(entry.path):
                for file in files:
                    if is_image_file(file):
                        count += 1
            print(f"{entry.name}: {count} images")

def main():
    parser = argparse.ArgumentParser(description="Count image files in each subdirectory of a main directory.")
    parser.add_argument("main_dir", help="Path to the main directory")
    args = parser.parse_args()
    count_images_in_subdirs(args.main_dir)

if __name__ == "__main__":
    main() 