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


SEED = 42
BATCH_SIZE = 32
EPOCHS = 12

HEAD_LR = 5e-4
BACKBONE_LR = 5e-5
WEIGHT_DECAY = 1e-4

PATIENCE = 4

CHECKPOINT_PATH = "models/resnet50_stage1_best.pth"
OUTPUT_CHECKPOINT = "models/resnet50_stage2_best.pth"
HISTORY_PATH = "results/resnet50_stage2_training_history.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(train_loader, num_classes, device):
    labels = []

    for _, batch_labels in train_loader:
        labels.extend(batch_labels.numpy().tolist())

    counts = np.bincount(labels, minlength=num_classes)

    weights = len(labels) / (
        num_classes * np.maximum(counts, 1)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):

    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda"
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    return epoch_loss, accuracy, macro_f1, weighted_f1


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda"
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(all_labels, all_preds)

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    return epoch_loss, accuracy, macro_f1, weighted_f1


def main():

    set_seed(SEED)

    print("=" * 70)
    print("WASTEWISE RESNET50 STAGE 2 - FINE-TUNING")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print(f"Device: {device}")

    train_loader, val_loader, _, class_names = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    num_classes = len(class_names)

    print(f"Classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # ---------------------------------------------------------
    # Load Stage 1 model
    # ---------------------------------------------------------

    model = WasteResNet50(
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nLoaded Stage 1 checkpoint:")
    print(CHECKPOINT_PATH)

    # ---------------------------------------------------------
    # Unfreeze layer4 + classification head
    # ---------------------------------------------------------

    model.unfreeze_last_layers()

    model = model.to(device)

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader,
        num_classes,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ---------------------------------------------------------
    # Separate learning rates
    # ---------------------------------------------------------

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.startswith("backbone.layer4"):
            backbone_params.append(param)

        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR
            },
            {
                "params": head_params,
                "lr": HEAD_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\nStarting Stage 2 fine-tuning...")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc, train_macro_f1, train_weighted_f1 = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler
            )
        )

        val_loss, val_acc, val_macro_f1, val_weighted_f1 = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        scheduler.step(val_macro_f1)

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Train F1 {train_macro_f1:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Macro F1 {val_macro_f1:.4f} | "
            f"LR backbone {current_lr_backbone:.2e} | "
            f"LR head {current_lr_head:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "train_weighted_f1": train_weighted_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "backbone_lr": current_lr_backbone,
            "head_lr": current_lr_head
        })

        # -----------------------------------------------------
        # Save best model
        # -----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_macro_f1,
                    "class_names": class_names
                },
                OUTPUT_CHECKPOINT
            )

            print(
                f"  ✓ New best model saved "
                f"(Val Macro F1: {best_macro_f1:.4f})"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"\nEarly stopping at epoch {epoch}."
                )

                break

    # ---------------------------------------------------------
    # Save history
    # ---------------------------------------------------------

    Path(HISTORY_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 70)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved: {OUTPUT_CHECKPOINT}")
    print(f"History saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()