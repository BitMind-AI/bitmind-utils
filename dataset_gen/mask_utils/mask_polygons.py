import os
import numpy as np
import cv2
import json
import concurrent.futures
import argparse

# CONFIGURATION
LOCAL_DATASETS_ROOT = "/workspace/bitmind-utils/dataset_gen/test_data/synthetic_images/dreamshaper-8-inpainting"  # or the parent directory containing all datasets
S3_BUCKET = "subnet-34-storage"
S3_PREFIX = "semisynthetics"  # or whatever your S3 prefix is

def mask_to_polygons(mask_path):
    mask = np.load(mask_path)
    # If mask is 3D (e.g., (256, 256, 1)), squeeze to 2D
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask, axis=2)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # Mask is 0/1, scale to 0/255 for visualization/contours
    if mask.max() == 1:
        mask = mask * 255
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.squeeze().tolist()
        if len(polygon) > 0:
            # Ensure each polygon is a list of [x, y] pairs
            if isinstance(polygon[0], list):
                polygons.append(polygon)
            else:
                polygons.append([polygon])
    return polygons

def local_path_to_s3_uri(local_path):
    rel_path = os.path.relpath(local_path, LOCAL_DATASETS_ROOT)
    parts = rel_path.split(os.sep)
    dataset = os.path.basename(LOCAL_DATASETS_ROOT)
    if len(parts) >= 2:
        subdataset = parts[0]
        filename = parts[-1]
        s3_key = os.path.join(S3_PREFIX, dataset, subdataset, filename).replace("\\", "/")
    else:
        s3_key = os.path.join(S3_PREFIX, dataset, rel_path).replace("\\", "/")
    return f"s3://{S3_BUCKET}/{s3_key}"

def process_image_mask_pair(args_tuple):
    img_path, mask_path = args_tuple
    polygons = mask_to_polygons(mask_path)
    s3_uri = local_path_to_s3_uri(img_path)
    return {
        "image-uri": s3_uri,
        "polygons": polygons
    }

def process_dataset(dataset_root, workers=8):
    image_mask_pairs = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                mask_path = os.path.splitext(img_path)[0] + "_mask.npy"
                if not os.path.exists(mask_path):
                    continue
                image_mask_pairs.append((img_path, mask_path))
    dataset_json = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process_image_mask_pair, image_mask_pairs):
            dataset_json.append(result)
    return dataset_json

def main():
    parser = argparse.ArgumentParser(description="Generate dataset JSONs with polygons using workers.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()
    workers = args.workers
    # For each dataset (e.g., dreamshaper-8-inpainting, stable-diffusion-xl-1.0-inpainting-0.1, etc.)
    for dataset_name in os.listdir(LOCAL_DATASETS_ROOT):
        dataset_path = os.path.join(LOCAL_DATASETS_ROOT, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        print(f"Processing dataset: {dataset_name}")
        dataset_json = process_dataset(dataset_path, workers=workers)
        out_json_path = f"{dataset_name}.json"
        with open(out_json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)
        print(f"Wrote {len(dataset_json)} entries to {out_json_path}")

if __name__ == "__main__":
    main()