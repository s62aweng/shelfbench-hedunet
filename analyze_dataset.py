import os
import torch
import random

def analyze_folder(folder_path, sample_size=5):
    pt_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".pt")]
    if not pt_files:
        return None

    # Zufällige Auswahl von bis zu sample_size Dateien
    sample = random.sample(pt_files, min(sample_size, len(pt_files)))

    min_vals, max_vals = [], []
    example_size, channels = None, None

    for fn in sample:
        path = os.path.join(folder_path, fn)
        try:
            tensor = torch.load(path, map_location="cpu")
            if isinstance(tensor, dict):
                tensor = tensor.get("data", None) or next(iter(tensor.values()))
            size = tuple(tensor.shape)
            channels = size[0] if len(size) > 1 else 1
            example_size = size
            min_vals.append(float(tensor.min()))
            max_vals.append(float(tensor.max()))
        except Exception as e:
            print(f"Could not read {path}: {e}")

    return {
        "files": len(pt_files),
        "example_size": example_size,
        "channels": channels,
        "min_val": min(min_vals) if min_vals else None,
        "max_val": max(max_vals) if max_vals else None
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

if __name__ == "__main__":
    dataset_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench_256_Pt"
    log_path = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet/dataset_analysis.txt"
    walk_and_log(dataset_root, log_path)
    print(f"Analysis written to {log_path}")
