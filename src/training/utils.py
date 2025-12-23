# src/training/utils.py
import os
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS_DIR, NUM_CLASSES


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (preds == labels).sum().item()
            n_samples += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else None

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), all_probs


def save_confusion_matrix(cm, class_names, filename: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs(filename.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_training_curves(history: Dict, filename: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    os.makedirs(filename.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
