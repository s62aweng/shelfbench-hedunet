import os
import torch

def analyze_folder(folder_path):
    pt_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".pt")]
    if not pt_files:
        return None

    # Nur erste Datei laden
    path = os.path.join(folder_path, pt_files[0])
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        tensor = tensor.get("data", None) or next(iter(tensor.values()))

    size = tuple(tensor.shape)
    channels = size[0] if len(size) > 1 else 1
    min_val = float(tensor.min())
    max_val = float(tensor.max())

    return {
        "files": len(pt_files),
        "example_size": size,
        "channels": channels,
        "min_val": min_val,
        "max_val": max_val
    }

def walk_and_log(root_dir, log_file):
    with open(log_file, "w") as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            result = analyze_folder(dirpath)
            if result:
                f.write(f"\nDirectory: {dirpath}\n")
                f.write(f"  Files: {result['files']}\n")
                f.write(f"  Example size: {result['example_size']}\n")
                f.write(f"  Channels: {result['channels']}\n")
                f.write(f"  Min value: {result['min_val']:.4f}\n")
                f.write(f"  Max value: {result['max_val']:.4f}\n")

if __name__ == "__main__":
    dataset_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench_256_Pt"
    log_path = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet/dataset_analysis.txt"
    walk_and_log(dataset_root, log_path)
    print(f"Analysis written to {log_path}")
