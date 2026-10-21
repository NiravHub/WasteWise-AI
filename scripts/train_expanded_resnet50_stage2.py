from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from sklearn.metrics import precision_recall_fscore_support

from src.data.expanded_dataloader import build_dataloaders
from src.models.resnet50_model import WasteResNet50


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 32
EPOCHS = 20

BACKBONE_LR = 5e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4

STAGE1_CHECKPOINT = (
    ROOT / "models" / "expanded_resnet50_stage1_best.pth"
)

OUTPUT_CHECKPOINT = (
    ROOT / "models" / "expanded_resnet50_stage2_best.pth"
)

HISTORY_PATH = (
    ROOT / "results" / "metrics" / "expanded"
    / "expanded_resnet50_stage2_history.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("EXPANDED RESNET50 — STAGE 2 FINE-TUNING")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


train_loader, val_loader, test_loader, class_to_idx, class_weights = (
    build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
)

num_classes = len(class_to_idx)

print(f"Classes: {num_classes}")
print(f"Train: {len(train_loader.dataset)}")
print(f"Validation: {len(val_loader.dataset)}")
print(f"Test: {len(test_loader.dataset)}")


# ============================================================
# MODEL
# ============================================================

model = WasteResNet50(
    num_classes=num_classes,
    pretrained=False,
)

checkpoint = torch.load(
    STAGE1_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

print("Loaded Stage 1 checkpoint.")


# Freeze everything first
model.freeze_backbone()


# Unfreeze only layer4
for param in model.backbone.layer4.parameters():
    param.requires_grad = True


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

backbone_params = [
    p for p in model.backbone.layer4.parameters()
    if p.requires_grad
]

head_params = [
    p for p in model.backbone.fc.parameters()
    if p.requires_grad
]

optimizer = AdamW(
    [
        {
            "params": backbone_params,
            "lr": BACKBONE_LR,
        },
        {
            "params": head_params,
            "lr": HEAD_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
)


# ============================================================
# AMP
# ============================================================

scaler = GradScaler("cuda") if DEVICE.type == "cuda" else None


# ============================================================
# TRAINING
# ============================================================

best_macro_f1 = 0.0
best_epoch = 0

history = []

start_time = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    running_loss = 0.0
    train_samples = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        if DEVICE.type == "cuda":

            with autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)

        train_samples += labels.size(0)


    train_loss = running_loss / train_samples


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0
    val_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            if DEVICE.type == "cuda":

                with autocast("cuda"):

                    outputs = model(images)
                    loss = criterion(
                        outputs,
                        labels,
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            predictions = outputs.argmax(
                dim=1
            )

            val_loss_total += (
                loss.item() * labels.size(0)
            )

            val_samples += labels.size(0)

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )


    val_loss = val_loss_total / val_samples


    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    accuracy = sum(
        int(a == b)
        for a, b in zip(
            all_labels,
            all_predictions,
        )
    ) / len(all_labels)


    scheduler.step(macro_f1)


    current_lrs = [
        group["lr"]
        for group in optimizer.param_groups
    ]


    epoch_time = time.time() - epoch_start


    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_macro_precision": precision,
        "val_macro_recall": recall,
        "val_macro_f1": macro_f1,
        "backbone_lr": current_lrs[0],
        "head_lr": current_lrs[1],
        "epoch_time_seconds": epoch_time,
    }

    history.append(record)


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {accuracy:.4f} | "
        f"Macro F1 {macro_f1:.4f} | "
        f"LR {current_lrs}"
    )


    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "epoch": epoch,
                "val_macro_f1": macro_f1,
                "stage": 2,
            },
            OUTPUT_CHECKPOINT,
        )

        print(
            f"  -> New best checkpoint saved "
            f"(Macro F1 = {macro_f1:.4f})"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

total_time = time.time() - start_time

result = {
    "experiment": "expanded_resnet50_stage2",
    "stage1_checkpoint": str(STAGE1_CHECKPOINT),
    "best_checkpoint": str(OUTPUT_CHECKPOINT),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "backbone_lr": BACKBONE_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_macro_f1,
    "total_training_time_seconds": total_time,
    "num_classes": num_classes,
    "class_to_idx": class_to_idx,
    "history": history,
}

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 70)
print("STAGE 2 TRAINING COMPLETE")
print("=" * 70)
print(f"Best epoch: {best_epoch}")
print(
    f"Best validation Macro F1: "
    f"{best_macro_f1:.4f}"
)
print(f"Checkpoint: {OUTPUT_CHECKPOINT}")
print(f"History: {HISTORY_PATH}")
print("=" * 70)
