import os
import cv2
import numpy as np

# Pfad zu den Trainingsbildern
TRAIN_IMG_DIR = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png/train/images"

def compute_mean_std(img_dir, max_files=None):
    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    if max_files:
        files = files[:max_files]

    means, stds = [], []
    for fname in files:
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Skaliere auf [0,1] wie im Loader
        img = img.astype(np.float32) / 255.0
        means.append(img.mean())
        stds.append(img.std())

    dataset_mean = float(np.mean(means))
    dataset_std = float(np.mean(stds))
    return dataset_mean, dataset_std

if __name__ == "__main__":
    mean, std = compute_mean_std(TRAIN_IMG_DIR, max_files=None)  # oder z.B. 5000 für Testlauf
    print(f"Dataset mean: {mean:.6f}, std: {std:.6f}")
