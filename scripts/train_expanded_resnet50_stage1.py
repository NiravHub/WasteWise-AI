from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from src.data.expanded_dataloader import build_dataloaders
from src.models.resnet50_model import WasteResNet50

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "metrics" / "expanded"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4


def calculate_metrics(predictions, targets, num_classes):

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1)
    }


def evaluate(model, loader, criterion, num_classes):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda"
            ):
                outputs = model(images)
                loss = criterion(
                    outputs,
                    targets
                )

            total_loss += (
                loss.item() * images.size(0)
            )

            total_samples += images.size(0)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_targets.extend(
                targets.cpu().tolist()
            )

    metrics = calculate_metrics(
        all_predictions,
        all_targets,
        num_classes
    )

    metrics["loss"] = (
        total_loss / total_samples
    )

    return metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type="cuda",
            enabled=DEVICE.type == "cuda"
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * images.size(0)
        )

        total_samples += images.size(0)

    return running_loss / total_samples


def main():

    print("=" * 70)
    print("WasteWise AI - Expanded ResNet50 Stage 1")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("\nLoading datasets...")

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        class_weights
    ) = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    num_classes = len(class_to_idx)

    print(f"Classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset):,}")
    print(
        f"Validation: "
        f"{len(validation_loader.dataset):,}"
    )
    print(
        f"Test: "
        f"{len(test_loader.dataset):,}"
    )

    print("\nCreating ResNet50...")

    model = WasteResNet50(
        num_classes=num_classes,
        pretrained=True
    )

    model.freeze_backbone()

    model = model.to(DEVICE)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    class_weights = class_weights.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    scaler = GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda"
    )

    best_macro_f1 = -1.0
    best_epoch = -1

    history = []

    checkpoint_path = (
        MODEL_DIR /
        "expanded_resnet50_stage1_best.pth"
    )

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            num_classes
        )

        scheduler.step(
            validation_metrics["macro_f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation": validation_metrics,
            "learning_rate": float(current_lr),
            "seconds": float(elapsed)
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"Val Acc: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"Val Macro F1: "
            f"{validation_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if (
            validation_metrics["macro_f1"]
            > best_macro_f1
        ):

            best_macro_f1 = (
                validation_metrics["macro_f1"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "class_to_idx":
                        class_to_idx,

                    "num_classes":
                        num_classes,

                    "epoch":
                        epoch,

                    "validation_metrics":
                        validation_metrics,

                    "best_macro_f1":
                        best_macro_f1
                },
                checkpoint_path
            )

            print(
                f"  -> New best checkpoint "
                f"saved: Macro F1 "
                f"{best_macro_f1:.4f}"
            )

    history_path = (
        RESULT_DIR /
        "expanded_resnet50_stage1_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "experiment": "expanded_resnet50_stage1",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "best_epoch": best_epoch,
                "best_validation_macro_f1":
                    best_macro_f1,
                "class_to_idx": class_to_idx,
                "history": history
            },
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"History: {history_path}"
    )


if __name__ == "__main__":
    main()
