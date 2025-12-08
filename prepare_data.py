#!/usr/bin/env python
from pathlib import Path
import rasterio as rio
import numpy as np
import torch
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import os

# === Konfiguration ===
PATCHSIZE  = 256   # Patchgröße in Pixeln
MAX_NODATA = 0.2   # maximaler Anteil Nodata pro Patch
SAVE_PNG   = False # False = nur .pt, True = zusätzlich PNG
TEST_SIZE  = 0.2   # Anteil Testdaten
VAL_SIZE   = 0.1   # Anteil Validierungsdaten
USE_FIXED_SPLIT = True

# === Dynamic Output-Path ===
BASE_DIR   = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench")
datatype   = "PNG" if SAVE_PNG else "Pt"
OUTPUT_DIR = BASE_DIR / f"Shelf-Bench_{PATCHSIZE}_{datatype}"

# Log directory
LOG_DIR = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet/logs/preprocessing")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create folder structure
for split in ["train","val","test"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "masks").mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        (OUTPUT_DIR / split / "images_png").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "masks_png").mkdir(parents=True, exist_ok=True)

# === Helfer: Formate vereinheitlichen ===
def ensure_chw_np(arr):
    """Bringt beliebige 2D/3D Arrays robust ins Format (C, H, W)."""
    if arr.ndim == 2:  # (H, W)
        return np.expand_dims(arr, 0)
    if arr.ndim == 3:
        c, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
        # Kanal vorne?
        if c in (1, 2, 3, 4):
            return arr
        # Kanal hinten?
        if arr.shape[-1] in (1, 2, 3, 4):
            return np.transpose(arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # Kanal in der Mitte?
        if arr.shape[1] in (1, 2, 3, 4):
            return np.transpose(arr, (1, 0, 2))  # (H, C, W) -> (C, H, W)
    raise ValueError(f"Unexpected shape {arr.shape} for ensure_chw_np")

def normalize_patch(patch):
    """Lokale Normalisierung + Percentile Clipping (pro Kanal)."""
    # patch: (C, H, W)
    out = np.empty_like(patch, dtype=np.uint8)
    for c in range(patch.shape[0]):
        band = patch[c]
        band = 10 * np.log10(np.clip(band, 1e-6, None))
        vmin, vmax = np.percentile(band, (2, 98))
        band = np.clip(band, vmin, vmax)
        out[c] = (255 * (band - vmin) / (vmax - vmin + 1e-6)).astype(np.uint8)
    return out

def visualize_tensor(filename, out_png):
    full_path = os.path.abspath(filename)
    tensor = torch.load(filename, map_location="cpu")

    plt.figure(figsize=(12,5))

    # Bilddarstellung links
    plt.subplot(1,2,1)
    if tensor.ndim == 2:
        plt.imshow(tensor.numpy(), cmap="gray")
    elif tensor.ndim == 3:
        plt.imshow(tensor[0].numpy(), cmap="gray")
    else:
        raise ValueError("Tensor hat unerwartete Dimensionen")
    plt.colorbar()
    plt.title("Visualization")

    # Histogramm oder Säulendiagramm rechts
    plt.subplot(1,2,2)
    if tensor.dtype == torch.bool:
        values, counts = torch.unique(tensor, return_counts=True)
        labels = ["False", "True"]
        plt.bar(labels, counts.numpy(), color="steelblue", edgecolor="black")
        plt.title("Verteilung von True/False")
        plt.ylabel("Häufigkeit")
    else:
        plt.hist(tensor.numpy().ravel(), bins=100, color="steelblue", edgecolor="black")
        plt.title("Histogram of values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def pad_image(img, patch_size):
    """Padding für (C, H, W)"""
    assert img.ndim == 3, f"pad_image erwartet (C,H,W), bekam {img.shape}"
    h, w = img.shape[1], img.shape[2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    return np.pad(img, ((0,0),(0,pad_h),(0,pad_w)), mode="constant")

def extract_patches(scene, mask, patch_size):
    """Sliding-Window Patch Extraction für (C, H, W)"""
    h, w = scene.shape[1], scene.shape[2]
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            s_patch = scene[:, y:y+patch_size, x:x+patch_size]
            m_patch = mask[:, y:y+patch_size, x:x+patchsize]
            # Filter: zu viel Hintergrund verwerfen
            if np.mean(s_patch == 0) < MAX_NODATA:
                patches.append((s_patch, m_patch, x, y))
    return patches

def save_patch(split, s_patch, m_patch, base_name, idx):
    """Speichert einheitlich: images float32 (C,H,W), masks bool (1,H,W)."""
    # Bild als float32
    img_tensor = torch.from_numpy(s_patch).float()
    # Maske auf 1 Kanal reduzieren und in bool konvertieren
    if m_patch.shape[0] != 1:
        m_patch = m_patch[:1]  # nur erstes Maskenband
    mask_tensor = torch.from_numpy(m_patch.astype(np.uint8)).to(torch.bool)

    torch.save(img_tensor, OUTPUT_DIR / split / "images" / f"{base_name}_{idx}.pt")
    torch.save(mask_tensor, OUTPUT_DIR / split / "masks" / f"{base_name}_{idx}.pt")

    if SAVE_PNG:
        # Images: (C,H,W) -> (H,W,C), danach normalisieren
        png_img = np.moveaxis(normalize_patch(s_patch), 0, -1)
        cv2.imwrite(str(OUTPUT_DIR / split / "images_png" / f"{base_name}_{idx}.png"), png_img)
        # Maske: erstes Band auf [0,255]
        cv2.imwrite(str(OUTPUT_DIR / split / "masks_png" / f"{base_name}_{idx}.png"),
                    (m_patch[0].astype(np.uint8) * 255))

def process_scene(scene_path, split):
    """Komplette Verarbeitung einer Szene (erzwingt (C,H,W) vor allem)."""
    mask_path = scene_path.parent.parent / 'masks' / scene_path.name
    assert mask_path.exists(), f"{mask_path} doesn't exist!"

    # Szene lesen
    with rio.open(scene_path) as raster:
        scene_raw = raster.read()
        # NoData-Werte abfangen
        if raster.nodata is not None:
            scene_raw = np.where(scene_raw == raster.nodata, 0, scene_raw)
        # harte Variante: alles unter -1000 auf 0 setzen
        scene_raw[scene_raw < -1000] = 0

    # Maske lesen
    with rio.open(mask_path) as raster:
        mask_raw = raster.read()

    # Einheitliches Format erzwingen
    scene = ensure_chw_np(scene_raw)
    mask  = ensure_chw_np(mask_raw)

    # Immer nur erstes Band (Graustufen)
    scene = scene[:1]
    mask  = mask[:1]

    # Padding
    scene = pad_image(scene, PATCHSIZE)
    mask  = pad_image(mask, PATCHSIZE)

    # Patches extrahieren
    patches = extract_patches(scene, mask, PATCHSIZE)
    for idx, (s_patch, m_patch, x, y) in enumerate(patches):
        # Normalisierung auch für .pt
        norm_patch = normalize_patch(s_patch)
        img_tensor = torch.from_numpy(norm_patch).float()
        mask_tensor = torch.from_numpy(m_patch.astype(np.uint8)).to(torch.bool)

        torch.save(img_tensor, OUTPUT_DIR / split / "images" / f"{scene_path.stem}_{idx}.pt")
        torch.save(mask_tensor, OUTPUT_DIR / split / "masks" / f"{scene_path.stem}_{idx}.pt")

    assert mask_path.exists(), f"{mask_path} doesn't exist!"

    # Szene lesen
    with rio.open(scene_path) as raster:
        scene_raw = raster.read()  # (Bands, H, W) oder anderes
    # Maske lesen
    with rio.open(mask_path) as raster:
        mask_raw = raster.read()   # (Bands, H, W) oder anderes

    # Einheitliches Format erzwingen
    scene = ensure_chw_np(scene_raw)
    mask  = ensure_chw_np(mask_raw)

    # Falls du IMMER Graustufen willst: nur erstes Band der Szene
    # (dein Text sagt "Images are greyscale", der Code nahm bisher 2 Bänder)
    scene = scene[:1]  # -> (1, H, W)

    # Maske nur erstes Band und binär (0/1) lassen
    mask = mask[:1]    # -> (1, H, W)

    # Padding
    scene = pad_image(scene, PATCHSIZE)
    mask = pad_image(mask, PATCHSIZE)

    # Patches extrahieren und speichern
    patches = extract_patches(scene, mask, PATCHSIZE)
    for idx, (s_patch, m_patch, x, y) in enumerate(patches):
        save_patch(split, s_patch, m_patch, scene_path.stem, idx)

# === Main ===
if __name__ == '__main__':
    satellites = ["ERS", "Envisat", "Sentinel-1"]
    all_train, all_val, all_test = [], [], []

    splits_path = Path(__file__).parent / "splits.txt"
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_names = set(splits["train"])
    val_names   = set(splits["val"])
    test_names  = set(splits["test"])

    for sat in satellites:
        input_dir = BASE_DIR / "Shelf-Bench-tifs" / sat / "scenes"
        scenes = list(input_dir.glob("*.tif"))

        # Testdaten liegen in eigenen Unterordnern
        if sat == "ERS":
            test_dir = BASE_DIR / "Shelf-Bench-tifs" / sat / "test_ers" / "scenes"
        elif sat == "Envisat":
            test_dir = BASE_DIR / "Shelf-Bench-tifs" / sat / "test_envisat" / "scenes"
        elif sat == "Sentinel-1":
            test_dir = BASE_DIR / "Shelf-Bench-tifs" / sat / "test_s1" / "scenes"
        else:
            test_dir = None

        test_scenes_all = list(test_dir.glob("*.tif")) if test_dir and test_dir.exists() else []

        if USE_FIXED_SPLIT:
            train_scenes = [s for s in scenes if s.stem in train_names]
            val_scenes   = [s for s in scenes if s.stem in val_names]
            test_scenes  = [s for s in test_scenes_all if s.stem in test_names]
        else:
            train_scenes, test_scenes = train_test_split(scenes, test_size=TEST_SIZE, random_state=42)
            train_scenes, val_scenes  = train_test_split(train_scenes, test_size=VAL_SIZE, random_state=42)

        # Verarbeitung für diesen Satelliten
        for scene in tqdm(train_scenes, desc=f"{sat} Train scenes"):
            process_scene(scene, "train")
        for scene in tqdm(val_scenes, desc=f"{sat} Val scenes"):
            process_scene(scene, "val")
        for scene in tqdm(test_scenes, desc=f"{sat} Test scenes"):
            process_scene(scene, "test")

        all_train.extend(train_scenes)
        all_val.extend(val_scenes)
        all_test.extend(test_scenes)

    # === Sanity-Check: Formen der gespeicherten Patches prüfen ===
    for split in ["train","val","test"]:
        img_files = list((OUTPUT_DIR / split / "images").glob("*.pt"))[:50]
        mask_files = list((OUTPUT_DIR / split / "masks").glob("*.pt"))[:50]
        for f_img, f_mask in zip(img_files, mask_files):
            img = torch.load(f_img, map_location="cpu")
            msk = torch.load(f_mask, map_location="cpu")
            assert img.ndim == 3 and img.shape[0] in (1,3), f"Bad image shape {img.shape} in {f_img}"
            assert msk.ndim == 3 and msk.shape[0] == 1 and msk.dtype == torch.bool, f"Bad mask {msk.shape}/{msk.dtype} in {f_mask}"

    # === Logging ===
    log_file = LOG_DIR / "preprocessing_log.txt"
    with open(log_file, "w") as f:
        f.write("Preprocessing Report\n")
        f.write("====================\n")
        f.write(f"PATCHSIZE: {PATCHSIZE}\n")
        f.write(f"MAX_NODATA: {MAX_NODATA}\n")
        f.write(f"SAVE_PNG: {SAVE_PNG}\n")
        f.write(f"TEST_SIZE: {TEST_SIZE}, VAL_SIZE: {VAL_SIZE}\n")
        f.write(f"USE_FIXED_SPLIT: {USE_FIXED_SPLIT}\n")
        f.write(f"Train scenes: {len(all_train)}\n")
        f.write(f"Val scenes: {len(all_val)}\n")
        f.write(f"Test scenes: {len(all_test)}\n")

    # === Beispielplots ===
    example_img = OUTPUT_DIR / "val" / "images" / "ERS_20100520_VV_142439_90.pt"
    example_mask = OUTPUT_DIR / "val" / "masks" / "ERS_20100520_VV_142439_90.pt"
    if example_img.exists():
        visualize_tensor(str(example_img), LOG_DIR / "example_scene.png")
    if example_mask.exists():
        visualize_tensor(str(example_mask), LOG_DIR / "example_mask.png")
