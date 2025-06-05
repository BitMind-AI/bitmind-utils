import os
import argparse
import numpy as np
import cv2
from PIL import Image
from bitmind.transforms import apply_random_augmentations
import concurrent.futures

# Supported image extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS

def process_and_save(args_tuple):
    input_path, mask_path, output_dir = args_tuple
    # Process image
    img = np.array(Image.open(input_path).convert('RGB'))
    img_aug, _, _ = apply_random_augmentations(img, (256, 256), level_probs={0: 1.0})
    img_out = Image.fromarray(img_aug)
    img_out.save(os.path.join(output_dir, os.path.basename(input_path)))

    # Process mask (assume .npy mask)
    if mask_path and os.path.exists(mask_path):
        mask = np.load(mask_path)
        print(f"Processing mask {mask_path}, shape: {mask.shape}, dtype: {mask.dtype}")
        if mask.size == 0:
            print(f"Warning: Mask {mask_path} is empty, skipping.")
            return
        if mask.ndim == 1:
            side = int(np.sqrt(mask.shape[0]))
            if side * side != mask.shape[0]:
                print(f"Warning: Mask {mask_path} shape {mask.shape} is not square, skipping.")
                return
            mask = mask.reshape((side, side))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            #print(f"Converted mask to grayscale, new shape: {mask.shape}, dtype: {mask.dtype}")
        if mask.ndim == 2:
            mask = mask[..., None]  # Make it (H, W, 1) so ComposeWithParams treats it as a single image
            #print(f"Expanded mask to 3D, new shape: {mask.shape}, dtype: {mask.dtype}")
        if mask.ndim != 3:
            print(f"ERROR: Mask {mask_path} is not 3D after all conversions, shape: {mask.shape}")
            return
        mask_aug, _, _ = apply_random_augmentations(mask, (256, 256), level_probs={0: 1.0})
        #print(f"DEBUG: mask_aug shape after augmentation: {getattr(mask_aug, 'shape', None)}")
        # Squeeze back to 2D before saving
        if mask_aug.ndim == 3 and mask_aug.shape[2] == 1:
            mask_aug = np.squeeze(mask_aug, axis=2)
        mask_out_path = os.path.join(output_dir, os.path.basename(mask_path))
        np.save(mask_out_path, mask_aug)

def main():
    parser = argparse.ArgumentParser(description="Apply level 0 augmentations (center crop + resize to 256x256) to images and masks.")
    parser.add_argument('input_dir', help='Input directory (recursively searched)')
    parser.add_argument('output_dir', help='Output directory (flat)')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather all image/mask pairs
    tasks = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if is_image_file(file):
                img_path = os.path.join(root, file)
                base, _ = os.path.splitext(file)
                mask_path = os.path.join(root, base + '_mask.npy')
                tasks.append((img_path, mask_path, args.output_dir))

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(process_and_save, tasks))

if __name__ == '__main__':
    main() 