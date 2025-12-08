import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pathlib import Path
from deep_learning.utils.data_aug import PTDataset 


class SingleTileDataset(Dataset):
    """Dataset für genau eine Image/Mask-Datei"""
    def __init__(self, image_file, mask_file):
        self.image_file = image_file
        self.mask_file = mask_file

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = torch.load(self.image_file)
        mask = torch.load(self.mask_file)
        return image, mask
    
class EmptyDataset(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

def get_batch(dataset_names, batch_size=1, augment=False, shuffle=False, names=["images","masks"]):
    # dataset_names darf String oder Liste sein
    if isinstance(dataset_names, str):
        folders = [dataset_names]
    elif isinstance(dataset_names, (list, tuple)):
        folders = dataset_names
    else:
        raise TypeError(f"dataset_names muss str oder list sein, nicht {type(dataset_names)}")

    loader = get_loader(folders, batch_size=batch_size, augment=augment, shuffle=shuffle, names=names)
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs, masks, *rest = batch
            return imgs, masks
        else:
            return batch
     # Falls Loader leer ist:
    raise ValueError(f"Kein Batch gefunden für {dataset_names}")

def _get_dataset(dataset, names=["images", "masks"], augment=True):
    base_path = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Shelf-Bench/Shelf-Bench_256_Pt")

    ds_path = base_path / dataset
    ds_path = Path(ds_path)

    # Fall 1: es ist ein Ordner (wie "val")
    if ds_path.is_dir():
        images_dir = ds_path / "images"
        masks_dir = ds_path / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            print(f"❌ Ordner fehlt: {images_dir} oder {masks_dir}")
            return EmptyDataset()
        return PTDataset(ds_path, names, augment=augment)

    # Fall 2: es ist eine einzelne Datei (wie "val/images/ERS_20100520_VV_142439_19.pt")
    elif ds_path.is_file():
        # Maskendatei muss denselben Namen haben, nur im masks-Ordner
        mask_file = ds_path.parent.parent / "masks" / ds_path.name
        if not mask_file.exists():
            print(f"❌ Maskendatei fehlt: {mask_file}")
            return EmptyDataset()
        return SingleTileDataset(ds_path, mask_file)

    else:
        print(f"❌ Pfad existiert nicht: {ds_path}")
        return EmptyDataset()

def get_loader(
    folders,
    batch_size,
    num_workers=2,
    augment=False,
    shuffle=False,
    names=["images", "masks"],
):
    datasets = [_get_dataset(ds, names=names, augment=augment) for ds in folders]
    datasets = [ds for ds in datasets if len(ds) > 0]  # leere rausfiltern
    concatenated = ConcatDataset(datasets)
    return DataLoader(
        concatenated,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )