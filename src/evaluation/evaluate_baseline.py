"""
Evaluate the trained WasteWise baseline CNN on the untouched test set.

Outputs:
- Test accuracy
- Macro / weighted precision, recall, F1
- Top-3 accuracy
- Classification report
- Confusion matrix
- Per-class metrics
- JSON metrics
- PNG confusion matrix
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.dataset import create_dataloaders, load_class_names
from src.models.cnn_baseline import WasteCNN


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
MODEL_PATH = Path("models/wastecnn_baseline_best.pth")
SPLIT_DIR = Path("data/splits")

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    return device


# ============================================================
# Top-K Accuracy
# ============================================================

def top_k_accuracy(logits, targets, k=3):
    top_k_predictions = torch.topk(logits, k=k, dim=1).indices

    correct = top_k_predictions.eq(targets.view(-1, 1))
    correct = correct.any(dim=1)

    return correct.sum().item()


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, test_loader, device, class_names):
    model.eval()

    all_targets = []
    all_predictions = []

    total_top3_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            total_top3_correct += top_k_accuracy(
                outputs,
                labels,
                k=min(3, len(class_names)),
            )

            total_samples += labels.size(0)

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(all_targets, all_predictions)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    top3_accuracy = total_top3_correct / total_samples

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    report_dict = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "top3_accuracy": float(top3_accuracy),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm,
    }


# ============================================================
# Save Confusion Matrix
# ============================================================

def save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(11, 9))

    plt.imshow(cm, interpolation="nearest")
    plt.title("WasteWise Baseline - Test Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        class_names,
    )

    # Add values inside cells
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    output_path = FIGURES_DIR / "baseline_confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Confusion matrix saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WASTEWISE BASELINE MODEL - TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    class_names = load_class_names(
        SPLIT_DIR / "class_names.txt"
    )

    print("\nClasses:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    _, _, test_loader, _ = create_dataloaders(
    train_csv=SPLIT_DIR / "train.csv",
    validation_csv=SPLIT_DIR / "validation.csv",
    test_csv=SPLIT_DIR / "test.csv",
    class_names_path=SPLIT_DIR / "class_names.txt",
    batch_size=BATCH_SIZE,
    num_workers=0,
)

    print(f"\nTest samples: {len(test_loader.dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WasteCNN(
        num_classes=len(class_names)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    # Support both direct state_dict and checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    print(f"Loaded model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nRunning test evaluation...")

    results = evaluate(
        model,
        test_loader,
        device,
        class_names,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Test Accuracy       : {results['accuracy']:.4f}")
    print(f"Macro Precision     : {results['precision_macro']:.4f}")
    print(f"Macro Recall        : {results['recall_macro']:.4f}")
    print(f"Macro F1            : {results['macro_f1']:.4f}")
    print(f"Weighted Precision  : {results['precision_weighted']:.4f}")
    print(f"Weighted Recall     : {results['recall_weighted']:.4f}")
    print(f"Weighted F1        : {results['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy      : {results['top3_accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 70)

    print(results["classification_report_text"])

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    metrics = {
        "model": "WasteCNN Baseline",
        "model_path": str(MODEL_PATH),
        "test_samples": len(test_loader.dataset),
        "num_classes": len(class_names),
        "class_names": class_names,
        "accuracy": results["accuracy"],
        "precision_macro": results["precision_macro"],
        "recall_macro": results["recall_macro"],
        "macro_f1": results["macro_f1"],
        "precision_weighted": results["precision_weighted"],
        "recall_weighted": results["recall_weighted"],
        "weighted_f1": results["weighted_f1"],
        "top3_accuracy": results["top3_accuracy"],
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    metrics_path = METRICS_DIR / "baseline_test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(f"\nMetrics saved: {metrics_path}")

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report_path = REPORTS_DIR / "baseline_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("WasteWise AI - Baseline CNN Test Evaluation\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Test Accuracy      : {results['accuracy']:.4f}\n")
        f.write(f"Macro Precision    : {results['precision_macro']:.4f}\n")
        f.write(f"Macro Recall       : {results['recall_macro']:.4f}\n")
        f.write(f"Macro F1           : {results['macro_f1']:.4f}\n")
        f.write(
            f"Weighted Precision : {results['precision_weighted']:.4f}\n"
        )
        f.write(
            f"Weighted Recall    : {results['recall_weighted']:.4f}\n"
        )
        f.write(
            f"Weighted F1        : {results['weighted_f1']:.4f}\n"
        )
        f.write(
            f"Top-3 Accuracy     : {results['top3_accuracy']:.4f}\n"
        )

        f.write("\n\n")
        f.write(results["classification_report_text"])

    print(f"Report saved: {report_path}")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(
        results["confusion_matrix"],
        class_names,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()