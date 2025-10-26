import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image

DEFAULT_MEAN = [0.3443, 0.3803, 0.4082]
DEFAULT_STD  = [0.1573, 0.1309, 0.1198]

def extract_mean_std(dataloader):
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
                    mean = torch.tensor(t.mean).view(-1,1,1)
                    std  = torch.tensor(t.std).view(-1,1,1)
                    return mean, std
    except Exception:
        pass
    return None, None

def unnormalize(img, mean, std):
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean).view(-1,1,1).to(img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std).view(-1,1,1).to(img.device)
    return img * std + mean

def pgd_attack_batch(model, images, labels, eps, alpha, iters, device, targeted=False, target_labels=None):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    labels = labels.to(device)

    adv_images = images.clone().detach()

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted=True")
            loss = nn.CrossEntropyLoss()(outputs, target_labels.to(device))
            
            grad_sign = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0].sign()
            adv_images = adv_images - alpha * grad_sign
        else:
            loss = nn.CrossEntropyLoss()(outputs, labels)
            grad_sign = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0].sign()
            adv_images = adv_images + alpha * grad_sign
        
        delta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        adv_images = torch.clamp(ori_images + delta, min=ori_images.min().item()-1.0, max=ori_images.max().item()+1.0).detach()

    return adv_images.detach()

def evaluate_pgd(model, dataloader, device, eps=0.005, alpha=None, iters=20,
                          out_dir="../data/adversarial/pgd", save_every=20, max_save=200,
                          targeted=False, target_class=None):
    """
    Run PGD attack on dataloader, evaluate and save adv examples.
    - eps, alpha, iters are in normalized input space (same as model inputs).
    - If alpha is None, use eps / iters.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean_t, std_t = extract_mean_std(dataloader)
    if mean_t is None or std_t is None:
        mean_t = torch.tensor(DEFAULT_MEAN).view(-1,1,1)
        std_t  = torch.tensor(DEFAULT_STD).view(-1,1,1)

    if alpha is None:
        alpha = eps / max(1, iters)

    criterion = nn.CrossEntropyLoss()
    total = 0
    clean_correct = 0
    adv_correct = 0
    clean_loss_total = 0.0
    adv_loss_total = 0.0
    saved = 0
    batch_idx = 0

    for images, labels in tqdm(dataloader, desc=f"PGD eps={eps}", leave=False):
        batch_idx += 1
        images = images.to(device)
        labels = labels.to(device)
        total += images.size(0)

        # clean eval
        with torch.no_grad():
            out = model(images)
            loss = criterion(out, labels)
            _, preds = out.max(1)
            clean_correct += (preds == labels).sum().item()
            clean_loss_total += loss.item() * images.size(0)

        # determine target labels for targeted attack
        target_labels = None
        if targeted:
            if target_class is None:
                # simple heuristic: target the second-most-likely class (could improve)
                with torch.no_grad():
                    probs = torch.softmax(out, dim=1)
                    target_labels = probs.argsort(dim=1)[:,-2]  # second-best
            else:
                target_labels = torch.full_like(labels, fill_value=int(target_class), device=device)

        adv_images = pgd_attack_batch(model, images, labels, eps=eps, alpha=alpha, iters=iters,
                                      device=device, targeted=targeted, target_labels=target_labels)

        # eval adv
        with torch.no_grad():
            out_adv = model(adv_images)
            loss_adv = criterion(out_adv, labels)
            _, preds_adv = out_adv.max(1)
            adv_correct += (preds_adv == labels).sum().item()
            adv_loss_total += loss_adv.item() * images.size(0)

        # save some unnormalized images periodically
        if (batch_idx % save_every == 0) and (max_save is None or saved < max_save):
            adv_cpu = adv_images.cpu()
            preds_cpu = preds_adv.cpu()
            labels_cpu = labels.cpu()
            for i in range(adv_cpu.size(0)):
                if max_save is not None and saved >= max_save:
                    break
                img = adv_cpu[i]
                img_unn = unnormalize(img.to(device), mean_t.to(device), std_t.to(device)).cpu()
                img_unn = torch.clamp(img_unn, 0.0, 1.0)
                fname = f"pgd_eps{eps:.4f}_idx{saved:06d}_true{int(labels_cpu[i])}_pred{int(preds_cpu[i])}.png"
                save_image(img_unn, os.path.join(out_dir, fname))
                saved += 1

    metrics = {
        "clean_acc": clean_correct / total,
        "adv_acc": adv_correct / total,
        "clean_loss": clean_loss_total / total,
        "adv_loss": adv_loss_total / total,
        "eps": eps,
        "saved": saved,
        "out_dir": os.path.abspath(out_dir)
    }
    return metrics
