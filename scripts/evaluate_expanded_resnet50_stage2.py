from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.data.expanded_dataloader import build_dataloaders
from src.models.resnet50_model import WasteResNet50


MODEL_PATH = ROOT / "models" / "expanded_resnet50_stage2_best.pth"
RESULT_DIR = ROOT / "results" / "metrics" / "expanded"
FIGURE_DIR = ROOT / "results" / "figures" / "expanded"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("EXPANDED RESNET50 STAGE 1 — TEST EVALUATION")
print("=" * 70)
print(f"Device: {DEVICE}")

train_loader, val_loader, test_loader, class_to_idx, class_weights = build_dataloaders(
    batch_size=32,
    num_workers=0,
)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print(f"Test samples: {len(test_loader.dataset)}")
print(f"Classes: {len(class_names)}")

model = WasteResNet50(
    num_classes=len(class_names),
    pretrained=False,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []
total_loss = 0.0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

test_loss = total_loss / total_samples

accuracy = accuracy_score(all_labels, all_predictions)

macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

metrics = {
    "experiment": "expanded_resnet50_stage2_test",
    "checkpoint": str(MODEL_PATH),
    "test_samples": total_samples,
    "num_classes": len(class_names),
    "classes": class_names,
    "test_loss": test_loss,
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
}

metrics_path = RESULT_DIR / "expanded_resnet50_stage2_test_metrics.json"
report_path = RESULT_DIR / "expanded_resnet50_stage2_test_classification_report.txt"
cm_path = RESULT_DIR / "expanded_resnet50_stage2_test_confusion_matrix.npy"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

import numpy as np
np.save(cm_path, cm)

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Test Loss:          {test_loss:.4f}")
print(f"Accuracy:           {accuracy:.4%}")
print(f"Macro Precision:    {macro_precision:.4%}")
print(f"Macro Recall:       {macro_recall:.4%}")
print(f"Macro F1:           {macro_f1:.4%}")
print(f"Weighted Precision: {weighted_precision:.4%}")
print(f"Weighted Recall:    {weighted_recall:.4%}")
print(f"Weighted F1:        {weighted_f1:.4%}")

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report)

print()
print(f"Metrics: {metrics_path}")
print(f"Report:  {report_path}")
print(f"CM:      {cm_path}")
print("=" * 70)
