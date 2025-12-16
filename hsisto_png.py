import os
import random
import cv2
import numpy as np
from collections import Counter

# Pfad zu den Masken (anpassen falls nötig)
MASK_DIR = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png-datapp/train/masks"

def print_mask_histograms(mask_dir, num_samples=10):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    sample_files = random.sample(mask_files, min(num_samples, len(mask_files)))

    for fname in sample_files:
        path = os.path.join(mask_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Could not read {fname}")
            continue

        # Zähle Pixelwerte
        counts = Counter(mask.flatten())
        total = mask.size

        print(f"\nHistogram for {fname} (total pixels: {total}):")
        # Nur die wichtigsten Werte ausgeben
        for val, cnt in sorted(counts.items()):
            perc = 100.0 * cnt / total
            print(f"  Value {val:3d}: {cnt:8d} pixels ({perc:5.2f}%)")

if __name__ == "__main__":
    print_mask_histograms(MASK_DIR, num_samples=10)
