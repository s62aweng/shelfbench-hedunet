import os
import shutil
from PIL import Image
import numpy as np

def copy_images_and_normalize_masks(src_root, dst_root):
    """
    Kopiert Images unverändert und normalisiert Masks (255->1).
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
                    # Maske laden und normalisieren
                    img = Image.open(src_path)
                    arr = np.array(img)
                    arr = (arr // 255).astype(np.uint8)
                    Image.fromarray(arr).save(dst_path)
                else:
                    # Images einfach kopieren
                    shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    src_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png"
    dst_root = "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench-tiles-png-masknorm"

    copy_images_and_normalize_masks(src_root, dst_root)
    print(f"Neuer Datensatz erstellt unter: {dst_root}")
