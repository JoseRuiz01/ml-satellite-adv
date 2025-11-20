# io.py
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
import tifffile

from .perceptual import apply_lab_perturbation
from .utils import unnormalize, DEFAULT_MEAN, DEFAULT_STD
try:
    from skimage import color as skcolor 
except Exception:
    skcolor = None


def save_adversarial_tif(
    adv_img_tensor,      
    index_in_batch: int,
    orig_path: str,
    mean_t,
    std_t,
    labels_cpu,
    out_dir: str,
    **kwargs
):
    """
    Fixed version that properly handles unnormalization.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Get single image
    if hasattr(adv_img_tensor, "ndim") and adv_img_tensor.ndim == 4:
        img = adv_img_tensor[index_in_batch]
    else:
        img = adv_img_tensor
    
    device = img.device if torch.is_tensor(img) else 'cpu'
    
    # Unnormalize: back to [0,1]
    img_unn = img * std_t.to(device) + mean_t.to(device)
    img_unn = torch.clamp(img_unn, 0, 1)
    
    # Convert to numpy (C, H, W)
    img_np = img_unn.cpu().numpy()
    
    # Convert to (H, W, C)
    if img_np.ndim == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Scale to [0, 255]
    img_final = (img_np * 255).astype(np.uint8)
    
    # Save
    base = Path(orig_path).stem
    fname_tif = f"{base}_true{int(labels_cpu[index_in_batch])}.tif"
    out_path = Path(out_dir) / fname_tif
    
    tifffile.imwrite(str(out_path), img_final)
    return str(out_path)

def load_tif_image(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load a multi-band .tif image as a numpy array (C,H,W), optionally normalized to [0,1].

    Args:
        path: Path to the .tif file.
        normalize: If True, scale values to [0,1] based on data range.

    Returns:
        np.ndarray: Image array in (C,H,W) format, dtype=float32 if normalized.
    """
    import tifffile
    img = tifffile.imread(path)  # Usually (H, W, C)
    if img.ndim == 2:
        img = img[..., np.newaxis]  # (H, W, 1)

    # Convert to (C,H,W)
    img = np.transpose(img, (2, 0, 1))

    if normalize:
        img = img.astype(np.float32)
        vmin, vmax = np.percentile(img, (1, 99.9))
        if vmax > vmin:
            img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        else:
            img = np.zeros_like(img, dtype=np.float32)

    return img
