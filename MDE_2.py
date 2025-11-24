"""
Code to calculate the MDE of ground truth labels with predicted fronts
We use our segmentation masks to derive the front line. Background and ocean are defined as 0 in masks, therefore scenes with backgrounds are pre-filtered out to avoid ice-background classifications
Code inspired by Gourmelen et al. (2022)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import os
import gc
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.spatial.distance import directed_hausdorff, cdist
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from data_processing.ice_data import IceDataset
from omegaconf import OmegaConf
from pathlib import Path
from paths import ROOT_GWS, ROOT_LOCAL

@dataclass
class ModelSpec:
    arch: str
    name: str
    ckpt_path: str

# ============ FILE FILTERING ============

def load_background_filters(base_dir: str) -> Dict[str, bool]:
    background_info_dir = os.path.join(base_dir, "background_scenes", "background_info")
    json_files = ["Envisat_backgrounds.json", "ERS_backgrounds.json", "Sentinel-1_backgrounds.json"]
    
    combined_filters = {}
    for json_file in json_files:
        json_path = os.path.join(background_info_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                combined_filters.update(json.load(f))
    
    return combined_filters

def get_valid_file_indices(dataset, background_filters: Dict[str, bool]) -> List[int]:
    valid_indices = []
    image_files = getattr(dataset, 'image_files', getattr(dataset, 'image_paths', []))
    
    basename_to_status = {os.path.basename(k): v for k, v in background_filters.items()}
    
    for idx, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        if basename in basename_to_status and not basename_to_status[basename]:
            valid_indices.append(idx)
    
    print(f"Dataset filtering: {len(valid_indices)} valid, {len(image_files) - len(valid_indices)} skipped")
    return valid_indices

def create_filtered_dataloader(dataset, valid_indices: List[int], batch_size: int = 8) -> DataLoader:
    filtered_dataset = Subset(dataset, valid_indices)
    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ============ MASK PROCESSING ============

def normalize_mask_to_2d(mask: np.ndarray) -> np.ndarray:
    mask = mask.squeeze()
    while mask.ndim > 2:
        mask = mask[0]
    return mask.astype(np.uint8)

def apply_morphological_filter(mask: np.ndarray, operation: str = 'opening', iterations: int = 2) -> np.ndarray:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask_binary = (mask > 0).astype(np.uint8)
    
    ops = {'erosion': binary_erosion, 'dilation': binary_dilation, 
           'opening': binary_opening, 'closing': binary_closing}
    
    return ops[operation](mask_binary, structure=structure, iterations=iterations).astype(np.uint8)

# ============ BOUNDARY EXTRACTION ============

def extract_boundary_contour(mask: np.ndarray, 
                            image: Optional[np.ndarray] = None,
                            morphological_iterations: int = 2,
                            min_contour_length: int = 50) -> Optional[np.ndarray]:
    mask_binary = (mask > 0.5).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.astype(np.uint8)
    
    if np.unique(mask_binary).size < 2:
        return None
    
    if morphological_iterations > 0:
        mask_binary = binary_opening(mask_binary, iterations=morphological_iterations).astype(np.uint8)
    
    ocean_mask = (mask_binary == 0)
    ocean_dilated = binary_dilation(ocean_mask, iterations=2).astype(np.uint8)
    ice_ocean_interface = mask_binary & ocean_dilated
    
    if not np.any(ice_ocean_interface):
        return None
    
    contours, _ = cv2.findContours((ice_ocean_interface * 255).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    valid_contours = [c for c in contours if len(c) >= min_contour_length]
    if not valid_contours:
        return None
    
    longest = max(valid_contours, key=len)
    boundary = longest.squeeze()
    
    if boundary.ndim == 1:
        boundary = boundary.reshape(1, -1)
    
    boundary = boundary[:, [1, 0]].astype(np.float32)
    
    if is_boundary_straight_line(boundary) or is_boundary_on_patch_edge(boundary, mask.shape):
        return None
    
    return boundary

def is_boundary_straight_line(boundary: np.ndarray, threshold: float = 0.95) -> bool:
    if boundary is None or len(boundary) < 10:
        return False
    
    try:
        coords = boundary.astype(np.float64)
        coords_centered = coords - np.mean(coords, axis=0)
        _, s, _ = np.linalg.svd(coords_centered, full_matrices=False)
        if len(s) > 1 and s[0] > 0:
            return (1 - s[1] / s[0]) > threshold
    except:
        pass
    
    rows, cols = boundary[:, 0], boundary[:, 1]
    return np.std(rows) < 3.0 or np.std(cols) < 3.0

def is_boundary_on_patch_edge(boundary: np.ndarray, image_shape: tuple, edge_threshold: int = 8) -> bool:
    if boundary is None or len(boundary) == 0:
        return False
    
    height, width = image_shape
    rows, cols = boundary[:, 0], boundary[:, 1]
    
    edge_ratios = [
        np.sum(rows <= edge_threshold) / len(boundary),
        np.sum(rows >= height - edge_threshold - 1) / len(boundary),
        np.sum(cols <= edge_threshold) / len(boundary),
        np.sum(cols >= width - edge_threshold - 1) / len(boundary)
    ]
    
    return max(edge_ratios) > 0.3

# ============ DISTANCE CALCULATION ============

def calculate_boundary_distance(pred_boundary: np.ndarray, gt_boundary: np.ndarray,
                               pixel_resolution_m: float, metric: str = 'mean') -> float:
    if pred_boundary is None or gt_boundary is None or len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return np.nan
    
    if metric == 'hausdorff':
        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        distance_pixels = max(d1, d2)
    else:
        distances = cdist(pred_boundary, gt_boundary, metric='euclidean')
        min_distances = distances.min(axis=1)
        distance_pixels = np.mean(min_distances) if metric == 'mean' else np.median(min_distances)
    
    return distance_pixels * pixel_resolution_m

def get_satellite_resolution(filename: str) -> float:
    filename_upper = filename.upper()
    if 'S1' in filename_upper:
        return 40.0
    elif 'ERS' in filename_upper or 'ENV' in filename_upper:
        return 30.0
    return 30.0

# ============ MODEL MANAGEMENT ============

def prepare_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    return device

def build_model_specs(base_path: str, ckpt_names: Dict[str, str]) -> List[ModelSpec]:
    specs = []
    for model_key, ckpt in ckpt_names.items():
        arch = model_key.split('_')[0]
        specs.append(ModelSpec(arch=arch, name=model_key, ckpt_path=os.path.join(base_path, arch, ckpt)))
    return specs

def load_models(model_specs: List[ModelSpec], cfg, device: torch.device) -> Dict[str, torch.nn.Module]:
    from omegaconf import OmegaConf
    from load_functions import load_model
    
    models = {}
    for spec in model_specs:
        try:
            cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg_copy.model.name = spec.arch
            
            model = load_model(cfg_copy, torch.device('cpu'))
            ckpt = torch.load(spec.ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            models[spec.name] = model
            del ckpt
            gc.collect()
            print(f"✓ {spec.name} loaded")
        except Exception as e:
            print(f"✗ Error loading {spec.name}: {e}")
    
    return models

def process_mask(masks: torch.Tensor) -> torch.Tensor:
    if masks.max() > 1:
        masks = masks / 255.0
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)
    return (1 - masks).long()

# ============ EVALUATION ============

def run_mde_evaluation(models: Dict[str, torch.nn.Module], test_loader: DataLoader, device: torch.device,
                      output_dir: str = './mde_results', filter_iterations: int = 2,
                      original_filenames: Optional[List[str]] = None) -> Dict[str, Dict]:
    all_results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    if original_filenames is None:
        original_filenames = getattr(test_loader.dataset, 'image_files', 
                                    [f"sample_{i}" for i in range(len(test_loader.dataset))])
    
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\nProcessing {model_idx}/{len(models)}: {model_name}")
        
        model = model.to(device)
        model.eval()
        
        all_distances = []
        all_valid_filenames = []
        skipped_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if len(batch) == 3:
                    images, masks, batch_filenames = batch
                else:
                    images, masks = batch
                    batch_start = batch_idx * test_loader.batch_size
                    batch_filenames = [original_filenames[batch_start + i] 
                                     for i in range(len(images)) if batch_start + i < len(original_filenames)]
                
                images_gpu = images.to(device)
                outputs = model(images_gpu)
                
                pred_masks_np = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                masks_np = process_mask(masks).cpu().numpy()
                images_np = images.cpu().numpy()
                
                if pred_masks_np.shape[1] == 2:
                    pred_masks_np = pred_masks_np[:, 1, :, :]
                elif pred_masks_np.shape[1] == 1:
                    pred_masks_np = pred_masks_np.squeeze(1)
                
                for i in range(len(batch_filenames)):
                    pred_mask = pred_masks_np[i] if pred_masks_np.ndim == 3 else pred_masks_np
                    gt_mask = masks_np[i] if masks_np.ndim == 3 else masks_np
                    filename = batch_filenames[i]
                    
                    pred_boundary = extract_boundary_contour(pred_mask, morphological_iterations=filter_iterations)
                    gt_boundary = extract_boundary_contour(gt_mask, morphological_iterations=filter_iterations)
                    
                    if pred_boundary is None or gt_boundary is None:
                        skipped_counts['no_boundary'] += 1
                        continue
                    
                    pixel_res = get_satellite_resolution(filename)
                    distance = calculate_boundary_distance(pred_boundary, gt_boundary, pixel_res, 'mean')
                    
                    if not np.isnan(distance):
                        all_distances.append(distance)
                        all_valid_filenames.append(filename)
                
                del outputs, pred_masks_np, masks_np, images_np, images_gpu
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        model = model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        if len(all_distances) > 0:
            results = {
                'overall': {
                    'mean': float(np.mean(all_distances)),
                    'std': float(np.std(all_distances)),
                    'median': float(np.median(all_distances)),
                    'n_samples': len(all_distances)
                }
            }
            all_results[model_name] = results
            
            print(f"  Mean: {results['overall']['mean']:.2f}m, Valid: {results['overall']['n_samples']}")
            
            with open(os.path.join(model_dir, 'detailed_results.csv'), 'w') as f:
                f.write("filename,distance_m\n")
                for fn, dist in zip(all_valid_filenames, all_distances):
                    f.write(f"{fn},{dist:.4f}\n")
        else:
            all_results[model_name] = {'overall': {'mean': float('nan'), 'n_samples': 0}}
    
    with open(os.path.join(output_dir, 'model_comparison.csv'), 'w') as f:
        f.write("model,mean_mde_m,std_m,median_m,n_valid\n")
        for model_name, results in all_results.items():
            if results['overall']['n_samples'] > 0:
                r = results['overall']
                f.write(f"{model_name},{r['mean']:.2f},{r['std']:.2f},{r['median']:.2f},{r['n_samples']}\n")
    
    print(f"\n✓ Results saved to {output_dir}")
    return all_results

# ============ MAIN ============

if __name__ == "__main__":
    
    parent_dir = ROOT_GWS / "benchmark_data_CB" / "ICE-BENCH"
    checkpoint_base = ROOT_GWS / "benchmark_data_CB" / "model_outputs"

    batch_size = 8
    device = prepare_device()
    
    cfg = OmegaConf.create({
        'model': {
            'name': 'Unet', 'encoder_name': 'resnet50', 'encoder_weights': 'imagenet',
            'in_channels': 1, 'classes': 2, 'img_size': 256
        }
    })
    
    iou_best_models = {
        "ViT_best_iou": "ViT_bs32_correct_labels_best_iou.pth",
        "Unet_best_iou": "Unet_bs32_correct_labels_latest_epoch.pth",
        "DeepLabV3_best_iou": "DeepLabV3_bs32_correct_labels_latest_epoch.pth",
        "FPN_best_iou": "FPN_bs32_correct_labels_best_iou.pth",
        "DinoV3_best_iou": "DinoV3_bs32_correct_labels_best_iou.pth",
    }
    
    model_specs = build_model_specs(checkpoint_base, iou_best_models)
    models = load_models(model_specs, cfg, device)
    
    test_datasets = IceDataset.create_test_datasets(parent_dir)
    test_dataset = list(test_datasets.values())[0]
    
    background_filters = load_background_filters(parent_dir)
    valid_indices = get_valid_file_indices(test_dataset, background_filters)
    test_loader = create_filtered_dataloader(test_dataset, valid_indices, batch_size)
    
    original_filenames = [os.path.basename(test_dataset.image_files[idx]) for idx in valid_indices]
    
    results = run_mde_evaluation(models, test_loader, device, './mde_results_filtered', 
                                filter_iterations=2, original_filenames=original_filenames)