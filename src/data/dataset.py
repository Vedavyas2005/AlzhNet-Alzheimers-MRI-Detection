# src/data/dataset.py
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pydicom
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.config import RAW_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, NUM_CLASSES


CLASS_MAP = {
    "NC": 0,   # Normal Control
    "MCI": 1,  # Mild Cognitive Impairment
    "AD": 2,   # Alzheimer's Disease
}


def is_dicom_file(path: Path) -> bool:
    return path.suffix.lower() == ".dcm"


def scan_dataset(root_dir: Path) -> Tuple[List[Path], List[int]]:
    """
    Walk through root_dir and collect paths to all .dcm files under AD, MCI, NC.
    """
    image_paths: List[Path] = []
    labels: List[int] = []

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    for class_name, class_idx in CLASS_MAP.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder '{class_name}' not found in {root_dir}")

        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if is_dicom_file(fpath):
                    image_paths.append(fpath)
                    labels.append(class_idx)

    if len(image_paths) == 0:
        raise RuntimeError(f"No DICOM (.dcm) files found in {root_dir}. Check dataset extraction.")

    return image_paths, labels


def dicom_to_pil(image_path: Path) -> Image.Image:
    """
    Read DICOM and convert to PIL grayscale image.
    """
    ds = pydicom.dcmread(str(image_path))
    arr = ds.pixel_array.astype(np.float32)

    # Min-max normalize to 0-255
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("L")  # single-channel
    return img


def get_transforms(train: bool = True):
    """
    torchvision transforms for train/test.
    """
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),              # [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


class AlzheimersMRIDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = dicom_to_pil(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Scan dataset, create train/val/test datasets and wrap in DataLoaders.
    Uses random_split but with fixed SEED for reproducibility.
    """
    from math import floor

    image_paths, labels = scan_dataset(RAW_DATA_DIR)

    # Shuffle in a reproducible way
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)

    image_paths = [image_paths[i] for i in indices]
    labels = [labels[i] for i in indices]

    n_total = len(image_paths)
    n_train = floor(n_total * TRAIN_RATIO)
    n_val = floor(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_paths = image_paths[:n_train]
    train_labels = labels[:n_train]

    val_paths = image_paths[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]

    test_paths = image_paths[n_train + n_val:]
    test_labels = labels[n_train + n_val:]

    train_dataset = AlzheimersMRIDataset(
        train_paths, train_labels, transform=get_transforms(train=True)
    )
    val_dataset = AlzheimersMRIDataset(
        val_paths, val_labels, transform=get_transforms(train=False)
    )
    test_dataset = AlzheimersMRIDataset(
        test_paths, test_labels, transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Total images: {n_total}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    print(f"Class distribution (NC, MCI, AD): {class_counts}")

    return train_loader, val_loader, test_loader
