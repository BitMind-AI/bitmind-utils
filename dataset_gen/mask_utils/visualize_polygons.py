import argparse
import numpy as np
import cv2
from mask_polygons import mask_to_polygons

def draw_polygons_on_mask(mask_path, polygons, output_path, color=(0,255,0), thickness=2):
    # Load the mask (assume grayscale or (H, W, 1))
    mask = np.load(mask_path)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.squeeze(mask, axis=2)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # Convert to 3-channel for color drawing
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Draw each polygon
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if pts.ndim == 2:
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask_rgb, [pts], isClosed=True, color=color, thickness=thickness)
    # Save the result
    cv2.imwrite(output_path, mask_rgb)
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize polygons on a mask and save as PNG.")
    parser.add_argument('mask_path', help='Path to the .npy mask file')
    parser.add_argument('output_path', help='Path to save the output PNG')
    args = parser.parse_args()

    polygons = mask_to_polygons(args.mask_path)
    draw_polygons_on_mask(args.mask_path, polygons, args.output_path)

if __name__ == '__main__':
    main()
