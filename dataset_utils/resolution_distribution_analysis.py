import os
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image
from math import gcd

IMAGES_DIR = 'images'
OUTPUT_FILE = 'resolution_distribution.txt'
TOLERANCE = 8  # pixels

"""
For each aspect ratio, snap each image's resolution to the most common resolution within ±8 pixels. For each aspect ratio, keep 1 bin per 0.5% of dataset frequency (no minimum). This ensures common aspect ratios retain more bins, while rare ones may have zero bins, and canonical resolutions reflect actual dataset values.
"""

def get_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None

def get_aspect_ratio(size):
    w, h = size
    if h == 0:
        return 'undefined'
    divisor = gcd(w, h)
    return f"{w // divisor}:{h // divisor}"

def snap_to_common(size, common_resolutions, tolerance=TOLERANCE):
    w, h = size
    for cw, ch in common_resolutions:
        if abs(w - cw) <= tolerance and abs(h - ch) <= tolerance:
            return (cw, ch)
    return size

def main():
    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
    with Pool(cpu_count()) as pool:
        resolutions = list(tqdm(pool.imap_unordered(get_image_resolution, image_files), total=len(image_files)))
    resolutions = [r for r in resolutions if r is not None]
    total = len(resolutions)

    # Count all unique resolutions per aspect ratio
    aspect_res_counter = defaultdict(Counter)
    aspect_ratio_counter = Counter()
    for res in resolutions:
        ar = get_aspect_ratio(res)
        aspect_res_counter[ar][res] += 1
        aspect_ratio_counter[ar] += 1

    # For each aspect ratio, get the most common resolutions
    snapped_resolutions = []
    for ar, res_counter in aspect_res_counter.items():
        # Sort by frequency
        common_resolutions = [res for res, _ in res_counter.most_common()]
        for res in res_counter.elements():
            snapped = snap_to_common(res, common_resolutions)
            snapped_resolutions.append((ar, snapped))

    # Count snapped resolutions per aspect ratio
    aspect_snapped_counter = defaultdict(Counter)
    for ar, res in snapped_resolutions:
        aspect_snapped_counter[ar][res] += 1

    canonical_resolutions = set()
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"Total images: {total}\n\n")
        f.write(f"Snapped to most common resolution within ±{TOLERANCE} pixels, bins per aspect ratio: 1 per 0.5% of dataset frequency (no minimum).\n\n")
        for ar, res_counter in aspect_snapped_counter.items():
            ar_total = aspect_ratio_counter[ar]
            percent = (ar_total / total) * 100
            num_bins = int(percent * 2)  # 1 bin per 0.5% of dataset frequency, no minimum
            if num_bins == 0:
                f.write(f"Aspect Ratio: {ar} (total: {ar_total}, {percent:.2f}% of dataset, bins kept: 0)\n\n")
                continue
            f.write(f"Aspect Ratio: {ar} (total: {ar_total}, {percent:.2f}% of dataset, bins kept: {num_bins})\n")
            bins_sorted = res_counter.most_common(num_bins)
            for b, count in bins_sorted:
                bin_percent = (count / ar_total) * 100
                f.write(f"  {b[0]}x{b[1]}: {count} ({bin_percent:.2f}%)\n")
                canonical_resolutions.add(b)
            f.write("\n")
        # Output canonical list for ResolutionSampler
        f.write("\nCanonical resolutions for ResolutionSampler:\n[")
        canonical_resolutions_sorted = sorted(canonical_resolutions)
        for i, res in enumerate(canonical_resolutions_sorted):
            f.write(f"{res}{',' if i < len(canonical_resolutions_sorted)-1 else ''}\n")
        f.write("]\n")

        # Print top 5 largest and smallest resolutions by frequency
        overall_counter = Counter(resolutions)
        # Largest: sort by area (w*h), then by frequency
        largest = sorted(overall_counter.items(), key=lambda x: (-x[0][0]*x[0][1], -x[1]))[:5]
        # Smallest: sort by area (w*h), then by frequency
        smallest = sorted(overall_counter.items(), key=lambda x: (x[0][0]*x[0][1], -x[1]))[:5]
        f.write("\nTop 5 most frequent largest resolutions:\n")
        for (w, h), count in largest:
            percent = (count / total) * 100
            f.write(f"  {w}x{h}: {count} ({percent:.2f}%)\n")
        f.write("\nTop 5 most frequent smallest resolutions:\n")
        for (w, h), count in smallest:
            percent = (count / total) * 100
            f.write(f"  {w}x{h}: {count} ({percent:.2f}%)\n")
    print(f"Canonical resolutions written to {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 