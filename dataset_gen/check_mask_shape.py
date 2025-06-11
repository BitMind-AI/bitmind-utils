import numpy as np
import sys

def check_mask_shape(mask_path):
    mask = np.load(mask_path)
    print(f"Mask path: {mask_path}")
    print(f"Shape: {mask.shape}")
    print(f"Dtype: {mask.dtype}")
    if mask.ndim == 2:
        print("This mask is 2D (H, W).")
    elif mask.ndim == 3:
        print(f"This mask is 3D (H, W, C) with {mask.shape[2]} channels.")
    else:
        print("Unexpected mask dimensions!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_mask_shape.py path/to/mask.npy")
    else:
        check_mask_shape(sys.argv[1])