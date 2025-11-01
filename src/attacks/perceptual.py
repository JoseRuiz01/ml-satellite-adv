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
    """
    Convert HxWx3 RGB in [0,1] to LAB. Uses skimage if available; otherwise raises.
    """
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
    raw_min: float,
    raw_max: float,
    smooth_sigma: float = 1.0,
    l_scale: float = 0.1,
    ab_scale: float = 0.7,
) -> np.ndarray:
    """
    Create perceptually-scaled RGB perturbation using LAB.

    Args:
        orig_rgb_raw: original image RGB bands in raw units (H,W,3)
        adv_rgb_scaled_01: adversarial RGB in [0,1] scaled to 0..1 range (H,W,3)
        raw_min, raw_max: scalar used to map between raw units and [0,1]
        smooth_sigma: gaussian smoothing sigma applied to delta LAB channels
        l_scale: scaling for L channel
        ab_scale: scaling for a/b channels

    Returns:
        rgb_recons_raw: HxW x3 array in raw dtype range (integers)
    """
    if skcolor is None:
        raise RuntimeError("skimage.color is required for perceptual operations")

    # Map orig to 0..1 (float)
    orig_rgb_01 = (orig_rgb_raw.astype(np.float32) - raw_min) / max(1.0, (raw_max - raw_min))
    orig_rgb_01 = np.clip(orig_rgb_01, 0.0, 1.0)
    adv_rgb_01 = np.clip(adv_rgb_scaled_01, 0.0, 1.0)

    lab_orig = rgb01_to_lab(orig_rgb_01)
    lab_adv = rgb01_to_lab(adv_rgb_01)

    delta_lab = lab_adv - lab_orig
    delta_lab[..., 0] *= l_scale
    delta_lab[..., 1:] *= ab_scale

    # smooth each LAB channel if possible
    if smooth_sigma > 0:
        if gaussian_filter is not None:
            for ch in range(delta_lab.shape[-1]):
                delta_lab[..., ch] = gaussian_filter(delta_lab[..., ch], sigma=smooth_sigma)
        else:
            # simple fallback: small local mean using convolution-like mean filter
            from scipy.signal import convolve2d  # may fail if scipy.signal not present
            kernel = np.ones((3, 3)) / 9.0
            for ch in range(delta_lab.shape[-1]):
                try:
                    delta_lab[..., ch] = convolve2d(delta_lab[..., ch], kernel, mode="same", boundary="symm")
                except Exception:
                    # if convolve unavailable, skip smoothing
                    pass

    rgb_recons = lab_to_rgb01(lab_orig + delta_lab)
    rgb_recons = np.clip(rgb_recons, 0.0, 1.0)
    return rgb_recons
