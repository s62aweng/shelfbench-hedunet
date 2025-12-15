import os
from PIL import Image
import numpy as np

def normalize_data(src_root, dst_root):
    """
    Kopiert Images mit Skalierung (0-255 -> 0-1) und normalisiert Masks (255->1).
    """
    for dirpath, _, filenames in os.walk(src_root):
        rel_path = os.path.relpath(dirpath, src_root)
        target_dir = os.path.join(dst_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for fn in filenames:
            if fn.lower().endswith(".png"):
                src_path = os.path.join(dirpath, fn)
                dst_path = os.path.join(target_dir, fn)

                if "masks" in dirpath.lower():
                    # Maske laden und normalisieren (255 -> 1)
                    img = Image.open(src_path)
                    arr = np.array(img)
                    arr = (arr // 255).astype(np.uint8)
                    Image.fromarray(arr).save(dst_path)
                else:
                    # Image laden, skalieren (0-255 -> 0-1), abspeichern
                    img = Image.open(src_path).convert("L")  # Graustufen
                    arr = np.array(img).astype(np.float32) / 255.0
                    # Werte liegen jetzt zwischen 0 und 1
                    # Beim Abspeichern wieder in 0-255 skalieren, sonst PNG verliert Info
                    arr_uint8 = (arr * 255).astype(np.uint8)
                    Image.fromarray(arr_uint8).save(dst_path)


if __name__ == "__main__":
    src_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png"
    dst_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png-norm"

    normalize_data(src_root, dst_root)
    print(f"Neuer Datensatz erstellt unter: {dst_root}")
