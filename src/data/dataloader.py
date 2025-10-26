import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def compute_mean_std(data_dir, sample_size=None):
    """
    Compute per-channel mean and std for a dataset stored in folders.
    """
    dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    n_samples = len(dataset) if sample_size is None else min(sample_size, len(dataset))
    mean = 0.0
    std = 0.0
    total_images = 0

    for i, (images, _) in enumerate(loader):
        if total_images >= n_samples:
            break
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean, std


def get_dataloaders(data_dir="data/raw", batch_size=32, val_split=0.15, test_split=0.15, seed=42):
    """
    Load the EuroSAT dataset organized in class folders.
    Splits into train/val/test and applies standard transforms.
    """

    mean, std = compute_mean_std(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, class_names
