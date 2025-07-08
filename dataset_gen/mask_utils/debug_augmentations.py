import os
import numpy as np
from glob import glob
from PIL import Image
from bitmind.transforms import apply_random_augmentations
from bitmind.generation.util.image import ensure_mask_3d

# Set your directory here
input_dir = "path/to/your/images"
output_dir = "augmented_debug/"
os.makedirs(output_dir, exist_ok=True)

max_pairs = 20  # Set your limit here

image_paths = sorted(glob(os.path.join(input_dir, "*.png")))
count = 0
for img_path in image_paths:
    if img_path.endswith("_mask.png"):
        continue  # skip mask images if present
    name = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(input_dir, f"{name}_mask.npy")
    if not os.path.exists(mask_path):
        print(f"Mask not found for {img_path}, skipping.")
        continue

    # Load image and mask
    img = np.array(Image.open(img_path))
    mask = np.load(mask_path)
    if mask.ndim == 2:
        mask = ensure_mask_3d(mask)

    # Save original image and mask as PNG for reference
    Image.fromarray(img).save(os.path.join(output_dir, f"{name}_orig.png"))
    orig_mask_img = np.squeeze(mask, axis=2) if mask.ndim == 3 and mask.shape[2] == 1 else mask
    Image.fromarray((orig_mask_img > 0).astype(np.uint8) * 255).save(
        os.path.join(output_dir, f"{name}_orig_mask.png")
    )

    # Apply augmentation (change level_probs as needed)
    aug_img, aug_mask, _, _ = apply_random_augmentations(
        img,
        target_image_size=img.shape[:2][::-1],
        mask=mask,
        level_probs={0: 1.0}  # deterministic, or adjust for random
    )

    # Squeeze and binarize mask for saving
    if aug_mask is not None and aug_mask.ndim == 3 and aug_mask.shape[2] == 1:
        aug_mask = np.squeeze(aug_mask, axis=2)
    aug_mask = (aug_mask > 0).astype(np.uint8) * 255

    # Save augmented image and mask as PNG
    Image.fromarray(aug_img).save(os.path.join(output_dir, f"{name}_aug.png"))
    Image.fromarray(aug_mask).save(os.path.join(output_dir, f"{name}_aug_mask.png"))

    print(f"Augmented and saved: {name}")

    count += 1
    if count >= max_pairs:
        print(f"Reached limit of {max_pairs} pairs.")
        break

print("Done! Check the 'augmented_debug' folder for results.")