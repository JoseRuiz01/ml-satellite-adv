# pgd.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# optional imports
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


@dataclass
class PGDConfig:
    eps: float = 0.005
    alpha: Optional[float] = None   
    iters: int = 50
    small_step_fraction: float = 0.2
    grad_mask_fraction: float = 0.25
    grad_blur_sigma: float = 1.0
    smooth_perturb_sigma: float = 1.0
    random_dither: bool = True
    dither_scale: float = 0.5
    device: Optional[torch.device] = None


def build_importance_mask(
    grad: torch.Tensor,
    keep_fraction: float = 0.3,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Build a soft per-pixel importance mask from gradients.

    Args:
        grad: gradient tensor of shape (B, C, H, W)
        keep_fraction: fraction of pixels to keep (highest magnitude)
        blur_sigma: gaussian blur sigma applied to binary mask (optional)
    """
    with torch.no_grad():
        magnitude = grad.abs().mean(dim=1)  # (B, H, W)
        B, H, W = magnitude.shape
        mask = torch.zeros_like(magnitude)

        k = max(1, int(H * W * keep_fraction))

        for i in range(B):
            flat = magnitude[i].view(-1)
            if k >= flat.numel():
                top_mask = torch.ones_like(flat)
            else:
                kth_idx = flat.numel() - k + 1
                thresh = torch.kthvalue(flat, kth_idx).values
                top_mask = (flat >= thresh).float()
            top_mask = top_mask.view(H, W).cpu().numpy()
            if gaussian_filter is not None and blur_sigma > 0:
                top_mask = gaussian_filter(top_mask, sigma=blur_sigma)
            else:
                kernel = np.ones((3, 3)) / 9.0
                top_mask = np.clip(
                    np.convolve(top_mask.ravel(), kernel.ravel(), mode="same"), 0, 1
                ).reshape(H, W)

            if top_mask.max() > 0:
                top_mask = top_mask / top_mask.max()

            mask[i] = torch.from_numpy(top_mask)

        mask = mask.unsqueeze(1).to(grad.device)  # (B,1,H,W)
        return mask


def compute_gradient_step(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool,
    target_labels: Optional[torch.Tensor],
    loss_fn=nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """
    Compute gradient with respect to inputs (no update) and return step direction.
    If targeted=True the step is negative gradient (move toward target).
    """
    imgs = imgs.clone().detach().requires_grad_(True)
    outputs = model(imgs)
    if targeted:
        if target_labels is None:
            raise ValueError("target_labels must be provided when targeted=True")
        loss = loss_fn(outputs, target_labels)
        grad = torch.autograd.grad(loss, imgs)[0]
        return -grad
    else:
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, imgs)[0]
        return grad


def _apply_gaussian_smooth_to_perturbation(pert: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply gaussian smoothing to perturbation stored as numpy (B,C,H,W).
    Fallback if scipy gaussian_filter is not available.
    """
    if gaussian_filter is None or sigma <= 0:
        return pert
    out = pert.copy()
    B, C, H, W = pert.shape
    for b in range(B):
        for c in range(C):
            out[b, c] = gaussian_filter(pert[b, c], sigma=sigma)
    return out


def pgd_attack_batch(
   model: nn.Module,
    images: torch.Tensor,  # Already normalized
    labels: torch.Tensor,
    mean: torch.Tensor,    # Normalization mean
    std: torch.Tensor,     # Normalization std
    eps: float = 0.01,     # Epsilon in PIXEL SPACE (0-255)
    alpha: float = None,
    iters: int = 20,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fixed PGD attack that works in pixel space to avoid artifacts.
    
    Key fix: Convert to pixel space [0,1], attack there, then normalize back.
    """
    device = images.device
    mean = mean.to(device)
    std = std.to(device)
    
    # 1. Unnormalize to pixel space [0, 1]
    images_pixel = images * std + mean
    images_pixel = torch.clamp(images_pixel, 0, 1)
    
    # 2. Convert epsilon to [0,1] space (from 0-255)
    eps_01 = eps / 255.0
    
    if alpha is None:
        alpha = eps_01 / 10
    
    # 3. Initialize adversarial images
    ori_pixel = images_pixel.clone().detach()
    adv_pixel = ori_pixel.clone().detach()
    
    loss_fn = nn.CrossEntropyLoss()
    
    for iteration in range(iters):
        # Normalize for model input
        adv_norm = (adv_pixel - mean) / std
        adv_norm.requires_grad_(True)
        
        # Forward pass
        outputs = model(adv_norm)
        
        # Compute loss
        if targeted and target_labels is not None:
            loss = -loss_fn(outputs, target_labels)
        else:
            loss = loss_fn(outputs, labels)
        
        # Backward pass
        grad = torch.autograd.grad(loss, adv_norm)[0]
        
        # Convert gradient back to pixel space
        grad_pixel = grad * std
        
        with torch.no_grad():
            # Update in pixel space
            adv_pixel = adv_pixel + alpha * grad_pixel.sign()
            
            # Project to epsilon ball in pixel space
            perturbation = adv_pixel - ori_pixel
            perturbation = torch.clamp(perturbation, -eps_01, eps_01)
            
            # Apply perturbation
            adv_pixel = ori_pixel + perturbation
            adv_pixel = torch.clamp(adv_pixel, 0, 1)
    
    # 4. Normalize back for return
    adv_norm = (adv_pixel - mean) / std
    return adv_norm.detach()



def smooth_perturbation(perturbation: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth perturbation to reduce high-frequency artifacts.
    """
    if sigma <= 0:
        return perturbation
    
    smoothed = perturbation.copy()
    for i in range(perturbation.shape[0]):  # Batch
        for j in range(perturbation.shape[1]):  # Channels
            smoothed[i, j] = gaussian_filter(perturbation[i, j], sigma=sigma)
    
    return smoothed
