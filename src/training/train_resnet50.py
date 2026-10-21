"""
WasteWise AI - ResNet50 Transfer Learning

Stage 1:
    - Load ImageNet-pretrained ResNet50
    - Freeze the complete backbone
    - Train only the custom classification head

Stage 2 will be performed separately after Stage 1.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import create_dataloaders
from src.models.resnet50_model import WasteResNet50


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 32
EPOCHS = 12

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 4

SPLIT_DIR = Path("data/splits")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Using GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print("CUDA not available. Using CPU.")

    return device


# ============================================================
# Class Weights
# ============================================================

def calculate_class_weights(train_loader, num_classes, device):

    class_counts = torch.zeros(num_classes)

    for _, labels in train_loader:

        for label in labels:
            class_counts[label] += 1

    total = class_counts.sum()

    weights = total / (
        num_classes * class_counts.clamp(min=1)
    )

    weights = weights.to(device)

    print("\nClass weights:")

    for i, weight in enumerate(weights):

        print(
            f"  Class {i}: {weight.item():.4f}"
        )

    return weights


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        all_targets.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WASTEWISE AI - RESNET50 STAGE 1 TRAINING")
    print("=" * 70)

    set_seed(SEED)

    device = get_device()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, validation_loader, test_loader, class_names = (
        create_dataloaders(
            train_csv=SPLIT_DIR / "train.csv",
            validation_csv=SPLIT_DIR / "validation.csv",
            test_csv=SPLIT_DIR / "test.csv",
            class_names_path=SPLIT_DIR / "class_names.txt",
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    num_classes = len(class_names)

    print("\nDataset:")
    print(f"  Classes           : {num_classes}")
    print(f"  Training samples  : {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(validation_loader.dataset)}")
    print(f"  Test samples      : {len(test_loader.dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WasteResNet50(
        num_classes=num_classes,
        pretrained=True,
    )

    # IMPORTANT:
    # Stage 1 = freeze ResNet backbone
    model.freeze_backbone()

    model = model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader,
        num_classes,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    best_model_path = (
        MODEL_DIR / "resnet50_stage1_best.pth"
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 70)

        (
            train_loss,
            train_accuracy,
            train_macro_f1,
            train_weighted_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
        ) = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nTrain Loss       : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : {train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1   : {train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted F1: {train_weighted_f1:.4f}"
        )

        print(
            f"\nVal Loss         : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy     : {val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1     : {val_macro_f1:.4f}"
        )

        print(
            f"Val Weighted F1  : {val_weighted_f1:.4f}"
        )

        print(
            f"Learning Rate    : {current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_macro_f1,
                "train_weighted_f1": train_weighted_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
                "learning_rate": current_lr,
            }
        )

        scheduler.step(val_macro_f1)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_macro_f1,
                    "class_names": class_names,
                },
                best_model_path,
            )

            print("\n✓ New best model saved.")

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR /
        "resnet50_stage1_training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESNET50 STAGE 1 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best model: {best_model_path}"
    )

    print(
        f"Training history: {history_path}"
    )


if __name__ == "__main__":
    main()