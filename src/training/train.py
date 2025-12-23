# src/training/train.py
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    REPORTS_DIR,
    SEED,
)
from src.data.dataset import create_dataloaders, CLASS_MAP
from src.models.alznet import build_model
from src.training.utils import (
    set_seed,
    train_one_epoch,
    evaluate,
    save_confusion_matrix,
    plot_training_curves,
)
from src.training.metrics import compute_classification_metrics


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders()
    model = build_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_val, y_val_pred, y_val_prob = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(BEST_MODEL_PATH.parent, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ New best model saved to {BEST_MODEL_PATH}")

    # Plot training curves
    plot_training_curves(history, REPORTS_DIR / "training_curves.png")

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_loss, test_acc, y_test, y_test_pred, y_test_prob = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob, num_classes=len(CLASS_MAP))
    print("\nTest Metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            continue
        print(f"{k}: {v}")

    # Save confusion matrix figure
    inv_class_map = {v: k for k, v in CLASS_MAP.items()}
    class_names = [inv_class_map[i] for i in range(len(inv_class_map))]
    cm = metrics["confusion_matrix"]
    save_confusion_matrix(cm, class_names, REPORTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
