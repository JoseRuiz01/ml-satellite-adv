import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
from src.data.dataloader import EuroSATDataset, compute_mean_std
from torchvision import transforms

# Fallback values (shouldn't be used because we compute mean/std from dataloader)
FALLBACK_MEAN = [0.3443, 0.3803, 0.4082]
FALLBACK_STD  = [0.1573, 0.1309, 0.1198]


def get_mean_std(data_dir, sample_size=2000, device="cpu"):
    """
    Build a base EuroSATDataset (with ToTensor) and compute mean/std using the function from dataloader.
    Returns mean and std as torch tensors shaped (C,1,1) on given device.
    """
    base_dataset = EuroSATDataset(data_dir, transform=transforms.ToTensor())
    mean, std = compute_mean_std(base_dataset, sample_size=sample_size)

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean)
    if not torch.is_tensor(std):
        std = torch.tensor(std)

    std[std == 0] = 1.0

    mean = mean.to(device).view(-1, 1, 1)
    std = std.to(device).view(-1, 1, 1)
    return mean, std


def unnormalize_tensor(img_tensor, mean, std):
    """
    img_tensor: (C,H,W) or (B,C,H,W), in normalized space.
    mean/std: torch tensors shaped (C,1,1) on same device.
    Returns tensor in [0,1] float range.
    """
    return img_tensor * std + mean


def fgsm_attack_batch(model, images, labels, epsilon, device):
    """
    FGSM for a single batch. Expects images to be normalized tensors on device and require_grad set.
    """
    images = images.clone().detach().to(device)
    images.requires_grad = True
    labels = labels.to(device)

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    adv_images = adv_images.detach()
    adv_images = torch.clamp(adv_images, images.min().item() - 1.0, images.max().item() + 1.0)
    return adv_images


def evaluate_fgsm(
    model,
    dataloader,
    data_dir,
    device,
    epsilon=0.01,
    out_dir="../data/adversarial/fgsm",
    save_every=10,
    max_save=None,
    mean_std_sample_size=2000
):
    """
    Run FGSM across dataloader and save adversarial images to disk using mean/std computed
    by compute_mean_std from the dataloader module.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    os.makedirs(out_dir, exist_ok=True)

    try:
        mean_t, std_t = get_mean_std(data_dir, sample_size=mean_std_sample_size, device=device)
    except Exception as e:
        print(f"Warning: failed to compute mean/std from data_dir ({e}), using fallback values.")
        mean_t = torch.tensor(FALLBACK_MEAN, device=device).view(-1,1,1)
        std_t  = torch.tensor(FALLBACK_STD, device=device).view(-1,1,1)

    total_samples = 0
    clean_correct = 0
    adv_correct = 0
    clean_loss_total = 0.0
    adv_loss_total = 0.0
    saved_count = 0
    batch_idx = 0

    for images, labels in tqdm(dataloader, desc=f"FGSM eps={epsilon}", leave=False):
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)
        total_samples += batch_size

        # Clean evaluation
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        clean_correct += (preds == labels).sum().item()
        clean_loss_total += loss.item() * batch_size

        # Generate adversarial examples (FGSM)
        adv_images = fgsm_attack_batch(model, images, labels, epsilon, device)

        # Evaluate adversarial
        outputs_adv = model(adv_images)
        loss_adv = criterion(outputs_adv, labels)
        _, preds_adv = torch.max(outputs_adv, 1)
        adv_correct += (preds_adv == labels).sum().item()
        adv_loss_total += loss_adv.item() * batch_size

        # Save some adversarial images to disk periodically
        if (batch_idx % save_every == 0) and (max_save is None or saved_count < max_save):
            adv_cpu = adv_images.cpu()
            preds_cpu = preds_adv.cpu()
            labels_cpu = labels.cpu()
            for i in range(adv_cpu.size(0)):
                if max_save is not None and saved_count >= max_save:
                    break
                img = adv_cpu[i]  
                img_unn = unnormalize_tensor(img.to(device), mean_t, std_t).cpu()
                img_unn = torch.clamp(img_unn, 0.0, 1.0)
                fname = f"adv_idx{saved_count:06d}.png"
                save_image(img_unn, os.path.join(out_dir, fname))
                saved_count += 1

    clean_acc = clean_correct / total_samples
    adv_acc = adv_correct / total_samples
    clean_loss = clean_loss_total / total_samples
    adv_loss = adv_loss_total / total_samples

    metrics = {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "clean_loss": clean_loss,
        "adv_loss": adv_loss,
        "eps": epsilon,
        "num_saved": saved_count,
        "out_dir": os.path.abspath(out_dir)
    }
    return metrics
