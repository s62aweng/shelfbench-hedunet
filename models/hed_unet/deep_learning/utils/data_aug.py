import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Wrapper für ToTensorV2, der .copy() erzwingt
class SafeToTensorV2(ToTensorV2):
    def apply(self, image, **params):
        return super().apply(image.copy(), **params)
    def apply_to_mask(self, mask, **params):
        return super().apply_to_mask(mask.copy(), **params)

class PTDataset(Dataset):
    """
    Random access Dataset für PyTorch-Tensoren, gespeichert wie:
        data/images/1.pt
        data/images/2.pt
        ...
        data/masks/1.pt
        data/masks/2.pt
        ...
    """
    def __init__(self, root, parts, augment=False, suffix='.pt'):
        self.root = Path(root)
        self.parts = parts
        self.augment = augment

        # Liste aller Dateien im ersten Part (z.B. images)
        first = self.root / parts[0]
        filenames = list(sorted([x.name for x in first.glob('*' + suffix)]))

        # Index: pro Datei ein Paar [image_path, mask_path]
        self.index = [[self.root / p / fname for p in parts] for fname in filenames]

        # Augmentierung + Normalisierung
        if self.augment:
            self.transform = A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2, p=1.0
                            ),
                            A.GaussNoise(p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.2),
                    A.Normalize(mean=0.3047126829624176,
                                std=0.32187142968177795),
                    SafeToTensorV2(),   # hier statt ToTensorV2
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=0.3047126829624176,
                                std=0.32187142968177795),
                    SafeToTensorV2(),   # hier statt ToTensorV2
                ]
            )

    def __getitem__(self, idx):
        files = self.index[idx]
        image = torch.load(files[0]).numpy()
        mask = torch.load(files[1]).numpy().astype("float32")

        augmented = self.transform(image=image.copy(), mask=mask.copy())

        image = augmented["image"].clone().contiguous()
        mask = augmented["mask"].clone().contiguous().float()

        # Einheitliches Format (C, H, W)
        def ensure_chw(t):
            if t.ndim == 2:              # (H, W)
                return t.unsqueeze(0)    # -> (1, H, W)
            elif t.ndim == 3:
                if t.shape[0] in [1, 3]:     # Kanal schon vorne
                    return t
                elif t.shape[-1] in [1, 3]:  # Kanal hinten
                    return t.permute(2, 0, 1)
            raise ValueError(f"Unexpected shape {t.shape}")

        image = ensure_chw(image)
        mask = ensure_chw(mask)

        return image, mask


    def __len__(self):
        return len(self.index)
