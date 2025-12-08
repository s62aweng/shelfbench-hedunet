import os
import torch

def analyze_pt_file(path):
    try:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            # falls deine .pt Dateien Dictionaries enthalten
            tensor = tensor.get("data", None) or next(iter(tensor.values()))
        size = tuple(tensor.shape)
        channels = size[0] if len(size) > 1 else 1
        min_val = float(tensor.min())
        max_val = float(tensor.max())
        return size, channels, min_val, max_val
    except Exception as e:
        return None, None, None, None

def walk_and_log(root_dir, log_file):
    with open(log_file, "w") as f:
        for dirpath, _, filenames in os.walk(root_dir):
            pt_files = [fn for fn in filenames if fn.endswith(".pt")]
            if not pt_files:
                continue
            f.write(f"\nDirectory: {dirpath}\n")
            for fn in pt_files:
                path = os.path.join(dirpath, fn)
                size, channels, min_val, max_val = analyze_pt_file(path)
                if size:
                    f.write(
                        f"  {fn}: size={size}, channels={channels}, "
                        f"min={min_val:.4f}, max={max_val:.4f}\n"
                    )
                else:
                    f.write(f"  {fn}: could not read\n")

if __name__ == "__main__":
    dataset_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench_256_Pt"
    log_path = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet/dataset_analysis.txt"
    walk_and_log(dataset_root, log_path)
    print(f"Analysis written to {log_path}")
