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
