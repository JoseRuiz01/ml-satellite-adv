# utils.py
from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np
import tifffile
import numpy as np
from skimage.transform import resize

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

def gdal_style_scale(img, src_min=0, src_max=2750, dst_min=1, dst_max=255):
    """Apply GDAL-style scaling and clipping."""
    img = img.astype(np.float32)
    img = np.clip(img, src_min, src_max)
    img = (img - src_min) / (src_max - src_min)
    img = img * (dst_max - dst_min) + dst_min
    img = np.clip(img, dst_min, dst_max)
    return (img / 255.0).astype(np.float32)  # normalize for display

def select_rgb_bands(img):
    """Selects RGB bands safely."""
    if img.ndim == 3 and img.shape[2] >= 4:
        # GDAL style: bands 4,3,2 (0-indexed)
        return img[:, :, [3,2,1]]
    elif img.ndim == 2:
        return np.stack([img]*3, axis=-1)
    else:
        return img[:, :, :3]

def scale_for_display(img, src_min=None, src_max=None):
    """Scale image to [0,1] for display."""
    img = img.astype(np.float32)
    if src_min is None:
        src_min = np.min(img)
    if src_max is None:
        src_max = np.max(img)
    if src_max - src_min < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - src_min) / (src_max - src_min)
    img = np.clip(img, 0, 1)
    return img

def load_and_prepare(path, size=(224,224), is_raw=True, is_adversarial=False):
    """
    Fixed version that handles both raw and adversarial images correctly.
    
    Args:
        path: Path to image
        size: Target size
        is_raw: True if loading raw satellite image
        is_adversarial: True if loading saved adversarial image
    """
    img = tifffile.imread(path)
    
    # If CHW, convert to HWC
    if img.ndim == 3 and img.shape[0] in [3,4,13]:
        img = np.transpose(img, (1,2,0))
    
    # Handle adversarial images (already processed uint8)
    if is_adversarial:
        # Adversarial images are already RGB uint8 [0, 255]
        if img.shape[2] > 3:
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
        
        # Resize
        img_rgb = resize(img_rgb, size, preserve_range=True, anti_aliasing=True)
        
        # Simply normalize to [0, 1]
        return img_rgb.astype(np.float32) / 255.0
    
    # Handle raw images (your original logic)
    else:
        # Select RGB bands
        img_rgb = select_rgb_bands(img)
        
        # Resize
        img_rgb = resize(img_rgb, size, preserve_range=True, anti_aliasing=True)
        
        # Apply appropriate scaling
        if is_raw:
            img_rgb = gdal_style_scale(img_rgb, src_min=0, src_max=2750)
        else:
            img_rgb = scale_for_display(img_rgb)
        
        return img_rgb

