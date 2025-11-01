# utils.py
from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np

DEFAULT_MEAN = [0.3443, 0.3803, 0.4082]
DEFAULT_STD = [0.1573, 0.1309, 0.1198]


def extract_mean_std(dataloader) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Attempt to extract Normalize(mean,std) from dataloader.dataset.transform.
    Returns (mean_tensor, std_tensor) in shape (C,1,1) or (None, None).
    """
    try:
        ds = dataloader.dataset
        if hasattr(ds, "dataset"):
            base = ds.dataset
        else:
            base = ds
        transform = getattr(base, "transform", None)
        if transform and hasattr(transform, "transforms"):
            for t in transform.transforms:
                if t.__class__.__name__ == "Normalize":
                    mean = torch.tensor(t.mean).view(-1, 1, 1)
                    std = torch.tensor(t.std).view(-1, 1, 1)
                    return mean, std
    except Exception:
        pass
    return None, None


def unnormalize(img: torch.Tensor, mean, std) -> torch.Tensor:
    """
    img: (C,H,W) or (B,C,H,W). mean/std can be tensor or list.
    returns image in same device as input.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean).view(-1, 1, 1).to(img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std).view(-1, 1, 1).to(img.device)
    return img * std + mean


def select_rgb_bands(img):
    if img.ndim == 3 and img.shape[2] >= 4:
        return img[:, :, [3,2,1]]
    if img.ndim == 2:
        return np.stack([img]*3, axis=-1)
    return img[:, :, :3]

def normalize_image(img, low=1, high=99):
    img = img.astype(np.float32)
    for c in range(img.shape[2]):
        p_low, p_high = np.percentile(img[:, :, c], (low, high))
        if p_high > p_low:
            img[:, :, c] = (img[:, :, c] - p_low) / (p_high - p_low)
        else:
            img[:, :, c] = 0
    return np.clip(img, 0, 1)