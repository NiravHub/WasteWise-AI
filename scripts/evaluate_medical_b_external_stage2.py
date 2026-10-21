from pathlib import Path
import sys
import json
import argparse
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.models.resnet50_model import WasteResNet50
from src.data.expanded_transforms import get_eval_transforms


class MedicalExternalDataset(Dataset):
    def __init__(self, records, class_to_idx):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = get_eval_transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]

        image = Image.open(item["path"]).convert("RGB")
        image = self.transform(image)

        return image, item["label"]


def build_class_mapping(train_csv):
    classes = set()

    with open(train_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            classes.add(row["canonical_class"])

    classes = sorted(classes)

    return {name: i for i, name in enumerate(classes)}, classes


def load_external_records(csv_path, class_to_idx):
    all_records = []
    supported_records = []
    unsupported_counts = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row["canonical_class"]
            path = Path(row["path"])

            all_records.append(row)

            if label not in class_to_idx:
                unsupported_counts[label] = (
                    unsupported_counts.get(label, 0) + 1
                )
                continue

            supported_records.append(
                {
                    "path": path,
                    "label": class_to_idx[label],
                    "class": label,
                }
            )

    return all_records, supported_records, unsupported_counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["stage1", "stage2"],
    )

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    external_csv = (
        ROOT
        / "data"
        / "splits_expanded"
        / "external_medical_test.csv"
    )

    train_csv = (
        ROOT
        / "data"
        / "splits_expanded"
        / "train.csv"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("WasteWise AI - Medical Waste Dataset B External Evaluation")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    class_to_idx, classes = build_class_mapping(
        train_csv
    )

    print(f"Model classes: {len(classes)}")

    all_records, supported_records, unsupported_counts = (
        load_external_records(
            external_csv,
            class_to_idx,
        )
    )

    print(f"Total Medical B images: {len(all_records)}")
    print(
        f"Supported-label images: {len(supported_records)}"
    )

    print()
    print("Unsupported Medical B classes:")

    if unsupported_counts:
        for label, count in sorted(
            unsupported_counts.items()
        ):
            print(
                f"  {label}: {count}"
            )
    else:
        print("  None")

    dataset = MedicalExternalDataset(
        supported_records,
        class_to_idx,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = WasteResNet50(
        num_classes=len(classes),
        pretrained=False,
    )

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
    )

    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):
        state_dict = checkpoint_data[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint_data

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []

    print()
    print("Running inference on supported classes...")

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(
            loader,
            start=1,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_targets.extend(
                targets.numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            if batch_idx % 10 == 0:
                print(
                    f"  Processed "
                    f"{min(batch_idx * 32, len(dataset))}"
                    f"/{len(dataset)}"
                )

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    supported_class_indices = sorted(
        set(y_true.tolist())
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=supported_class_indices,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=supported_class_indices,
            average="weighted",
            zero_division=0,
        )
    )

    supported_class_names = [
        classes[i]
        for i in supported_class_indices
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=supported_class_indices,
        target_names=supported_class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=supported_class_indices,
        target_names=supported_class_names,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=supported_class_indices,
    )

    metrics = {
        "experiment": (
            f"expanded_resnet50_"
            f"{args.stage}_medicalB_external"
        ),
        "checkpoint": str(checkpoint),
        "dataset": "Medical Waste 4.0 Dataset B",
        "split": "external_medical_test",
        "total_external_images": len(all_records),
        "supported_images": len(supported_records),
        "unsupported_images": (
            len(all_records)
            - len(supported_records)
        ),
        "unsupported_classes": unsupported_counts,
        "num_model_classes": len(classes),
        "evaluated_classes": supported_class_names,
        "accuracy_supported_classes": float(
            accuracy
        ),
        "macro_precision_supported_classes": float(
            precision_macro
        ),
        "macro_recall_supported_classes": float(
            recall_macro
        ),
        "macro_f1_supported_classes": float(
            f1_macro
        ),
        "weighted_precision_supported_classes": float(
            precision_weighted
        ),
        "weighted_recall_supported_classes": float(
            recall_weighted
        ),
        "weighted_f1_supported_classes": float(
            f1_weighted
        ),
        "classification_report": report_dict,
    }

    output_dir = (
        ROOT
        / "results"
        / "metrics"
        / "expanded"
        / "medicalB"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / (
            f"expanded_resnet50_"
            f"{args.stage}_medicalB_metrics.json"
        )
    )

    report_path = (
        output_dir
        / (
            f"expanded_resnet50_"
            f"{args.stage}_medicalB_"
            f"classification_report.txt"
        )
    )

    cm_path = (
        output_dir
        / (
            f"expanded_resnet50_"
            f"{args.stage}_medicalB_"
            f"confusion_matrix.npy"
        )
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    np.save(
        cm_path,
        cm,
    )

    print()
    print("=" * 70)
    print("MEDICAL B EXTERNAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Total Medical B images: {len(all_records)}"
    )

    print(
        f"Supported images evaluated: "
        f"{len(supported_records)}"
    )

    print(
        f"Unsupported images: "
        f"{len(all_records) - len(supported_records)}"
    )

    print(
        f"Accuracy:           {accuracy:.4%}"
    )

    print(
        f"Macro Precision:    "
        f"{precision_macro:.4%}"
    )

    print(
        f"Macro Recall:       "
        f"{recall_macro:.4%}"
    )

    print(
        f"Macro F1:           "
        f"{f1_macro:.4%}"
    )

    print(
        f"Weighted Precision: "
        f"{precision_weighted:.4%}"
    )

    print(
        f"Weighted Recall:    "
        f"{recall_weighted:.4%}"
    )

    print(
        f"Weighted F1:        "
        f"{f1_weighted:.4%}"
    )

    print()
    print("Unsupported classes:")

    for label, count in sorted(
        unsupported_counts.items()
    ):
        print(
            f"  {label}: {count}"
        )

    print()
    print("Classification Report:")
    print(report_text)

    print()
    print("Saved:")
    print(metrics_path)
    print(report_path)
    print(cm_path)


if __name__ == "__main__":
    main()
