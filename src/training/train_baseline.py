from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.data.dataset import create_dataloaders
from src.models.cnn_baseline import WasteCNN


# ============================================================
# WasteWise AI - Custom CNN Baseline Training
# ============================================================

SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 5

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        return device

    print("CUDA unavailable. Using CPU.")

    return torch.device("cpu")


# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

def calculate_class_weights(
    train_loader,
    num_classes,
    device,
):

    labels = train_loader.dataset.data["label"]

    class_names = train_loader.dataset.class_names

    counts = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    for index, class_name in enumerate(class_names):

        counts[index] = np.sum(
            labels == class_name
        )

    # Balanced inverse-frequency weights:
    #
    # weight_i = total / (num_classes * class_count_i)

    total = counts.sum()

    weights = total / (
        num_classes * counts
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print("\nClass weights:")

    for class_name, weight in zip(
        class_names,
        weights.cpu().tolist(),
    ):

        print(
            f"  {class_name:<15} "
            f"{weight:.4f}"
        )

    return weights


# ------------------------------------------------------------
# Training epoch
# ------------------------------------------------------------

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

    all_predictions = []
    all_labels = []

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

        # Automatic mixed precision
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
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    epoch_weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1,
        epoch_weighted_f1,
    )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

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
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    epoch_weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1,
        epoch_weighted_f1,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("WASTEWISE AI - CUSTOM CNN BASELINE")
    print("=" * 70)

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_loader,
        validation_loader,
        test_loader,
        class_names,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    num_classes = len(
        class_names
    )

    print(
        f"\nNumber of classes: "
        f"{num_classes}"
    )

    print(
        f"Training images: "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation images: "
        f"{len(validation_loader.dataset):,}"
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WasteCNN(
        num_classes=num_classes
    ).to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Mixed precision scaler
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "train_weighted_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "learning_rate": [],
    }

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    best_model_path = (
        MODEL_DIR /
        "wastecnn_baseline_best.pth"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Epoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )

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

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        scheduler.step(
            val_macro_f1
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["train_macro_f1"].append(
            train_macro_f1
        )

        history["train_weighted_f1"].append(
            train_weighted_f1
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_macro_f1"].append(
            val_macro_f1
        )

        history["val_weighted_f1"].append(
            val_weighted_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"\nTrain Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro F1   : "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted F1: "
            f"{train_weighted_f1:.4f}"
        )

        print(
            f"\nVal Loss         : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy     : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1     : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Val Weighted F1  : "
            f"{val_weighted_f1:.4f}"
        )

        print(
            f"Learning Rate    : "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = (
                val_macro_f1
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_names":
                        class_names,

                    "epoch":
                        epoch,

                    "val_macro_f1":
                        val_macro_f1,

                    "val_accuracy":
                        val_accuracy,
                },
                best_model_path,
            )

            print(
                "\n✓ New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement for "
                f"{epochs_without_improvement} "
                f"epoch(s)."
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR /
        "baseline_training_history.json"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest validation Macro F1: "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Best model: "
        f"{best_model_path}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )


if __name__ == "__main__":
    main()