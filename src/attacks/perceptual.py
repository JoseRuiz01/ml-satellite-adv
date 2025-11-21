from typing import Tuple, Optional
import numpy as np

try:
    from skimage import color as skcolor
except Exception:
    skcolor = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

def rgb01_to_lab(rgb01: np.ndarray) -> np.ndarray:
    if skcolor is None:
        raise RuntimeError("skimage.color is required for LAB conversions")
    return skcolor.rgb2lab(rgb01)


def lab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    if skcolor is None:
        raise RuntimeError("skimage.color is required for LAB conversions")
    return skcolor.lab2rgb(lab)

def apply_lab_perturbation(
    orig_rgb_raw: np.ndarray,
    adv_rgb_scaled_01: np.ndarray,
    smooth_sigma: float = 1.0,
    l_scale: float = 0.05,
    ab_scale: float = 0.4,
) -> np.ndarray:

    if skcolor is None:
        raise RuntimeError("skimage.color is required for perceptual operations")

    # Use actual min/max of the original RGB
    orig_min, orig_max = orig_rgb_raw.min(), orig_rgb_raw.max()
    scale = max(1.0, orig_max - orig_min)
    
    orig_rgb_01 = (orig_rgb_raw.astype(np.float32) - orig_min) / scale
    orig_rgb_01 = np.clip(orig_rgb_01, 0.0, 1.0)
    
    adv_rgb_01 = np.clip(adv_rgb_scaled_01, 0.0, 1.0)

    lab_orig = skcolor.rgb2lab(orig_rgb_01)
    lab_adv = skcolor.rgb2lab(adv_rgb_01)

    delta_lab = lab_adv - lab_orig
    delta_lab[..., 0] *= l_scale
    delta_lab[..., 1:] *= ab_scale

    if smooth_sigma > 0 and gaussian_filter is not None:
        for ch in range(delta_lab.shape[-1]):
            delta_lab[..., ch] = gaussian_filter(delta_lab[..., ch], sigma=smooth_sigma)

    rgb_recons = skcolor.lab2rgb(lab_orig + delta_lab)

    # Map back to original raw range
    rgb_recons = np.clip(rgb_recons, 0.0, 1.0)
    rgb_recons = rgb_recons * scale + orig_min

    return rgb_recons.astype(np.float32)

