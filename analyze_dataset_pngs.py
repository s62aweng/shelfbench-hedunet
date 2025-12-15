import os
import random
import torch
from PIL import Image
import numpy as np

def analyze_folder(folder_path, sample_size=20):
    png_files = [fn for fn in os.listdir(folder_path) if fn.lower().endswith(".png")]
    if not png_files:
        return None

    # Zufällige Auswahl von bis zu sample_size Dateien
    sample = random.sample(png_files, min(sample_size, len(png_files)))

    min_vals, max_vals, dtypes = [], [], []
    example_size, channels = None, None

    for fn in sample:
        path = os.path.join(folder_path, fn)
        try:
            img = Image.open(path)
            arr = np.array(img)
            tensor = torch.from_numpy(arr)

            size = tuple(tensor.shape)
            # Falls Graustufen: shape (H, W), sonst (H, W, C)
            if len(size) == 2:
                channels = 1
            else:
                channels = size[2]
            example_size = size

            min_vals.append(float(tensor.min()))
            max_vals.append(float(tensor.max()))
            dtypes.append(str(tensor.dtype))
        except Exception as e:
            print(f"Could not read {path}: {e}")

    return {
        "files": len(png_files),
        "example_size": example_size,
        "channels": channels,
        "min_val": min(min_vals) if min_vals else None,
        "max_val": max(max_vals) if max_vals else None,
        "dtypes": list(set(dtypes))  # unterschiedliche Datentypen im Sample
    }

def walk_and_log(root_dir, log_file):
    with open(log_file, "w") as f:
        for dirpath, _, filenames in os.walk(root_dir):
            result = analyze_folder(dirpath)
            if result:
                f.write(f"\nDirectory: {dirpath}\n")
                f.write(f"  Files: {result['files']}\n")
                f.write(f"  Example size: {result['example_size']}\n")
                f.write(f"  Channels: {result['channels']}\n")
                f.write(f"  Min value (sampled): {result['min_val']:.4f}\n")
                f.write(f"  Max value (sampled): {result['max_val']:.4f}\n")
                f.write(f"  Dtypes (sampled): {result['dtypes']}\n")

if __name__ == "__main__":
    dataset_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png-norm"
    log_path = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet/dataset_analysis.txt"
    walk_and_log(dataset_root, log_path)
    print(f"Analysis written to {log_path}")
