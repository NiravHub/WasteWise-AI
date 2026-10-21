from pathlib import Path
import sys
import json

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# WasteWise imports
# ---------------------------------------------------------------------
from src.models.resnet50_model import WasteResNet50


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CSV_PATH = ROOT / "data" / "splits_expanded" / "test.csv"
MODEL_PATH = ROOT / "models" / "expanded_resnet50_stage2_best.pth"
OOD_PATH = ROOT / "models" / "expanded_ood_calibration.json"


# ---------------------------------------------------------------------
# Canonical 19-class taxonomy
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "battery",
    "biological",
    "cable",
    "cardboard",
    "clothes",
    "electronic_device",
    "gauze",
    "general_trash",
    "glass",
    "medical_cap",
    "medical_glasses",
    "medical_gloves",
    "metal",
    "paper",
    "plastic",
    "shoe_cover",
    "shoes",
    "test_tube",
    "urine_bag",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ---------------------------------------------------------------------
# Production preprocessing
# ---------------------------------------------------------------------
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint


def load_ood_threshold():
    with open(OOD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data["threshold"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    print("=" * 80)
    print("WASTEWISE — EXPANDED HELD-OUT CLASSIFIER EVALUATION")
    print("=" * 80)

    print(f"CSV:    {CSV_PATH}")
    print(f"MODEL:  {MODEL_PATH}")
    print(f"OOD:    {OOD_PATH}")
    print(f"DEVICE: {DEVICE}")
    print()

    # -------------------------------------------------------------
    # Load manifest
    # -------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    print(f"Manifest rows: {len(df)}")

    required_columns = {
        "dataset",
        "canonical_class",
        "path",
        "sha256",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required CSV columns: {sorted(missing)}"
        )

    # -------------------------------------------------------------
    # Validate labels
    # -------------------------------------------------------------
    unknown_classes = sorted(
        set(df["canonical_class"]) - set(CLASS_NAMES)
    )

    if unknown_classes:
        raise RuntimeError(
            f"Unknown canonical classes: {unknown_classes}"
        )

    # -------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------
    missing_files = [
        path
        for path in df["path"].tolist()
        if not Path(path).exists()
    ]

    print(f"Missing image files: {len(missing_files)}")

    if missing_files:
        print("First missing files:")
        for path in missing_files[:10]:
            print("  ", path)

        raise RuntimeError(
            "Evaluation stopped because image files are missing."
        )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------
    model = WasteResNet50(
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)

    checkpoint = load_checkpoint(
        model,
        MODEL_PATH,
    )

    model.eval()

    print("Model loaded successfully.")

    # -------------------------------------------------------------
    # OOD threshold
    # -------------------------------------------------------------
    ood_threshold = load_ood_threshold()

    print(
        f"OOD threshold: {ood_threshold:.6f}"
    )

    # -------------------------------------------------------------
    # Evaluation containers
    # -------------------------------------------------------------
    y_true = []
    y_pred = []

    results = []

    inference_errors = []

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------
    with torch.no_grad():

        for index, row in df.iterrows():

            image_path = Path(row["path"])
            true_class = row["canonical_class"]

            try:

                image = Image.open(image_path).convert("RGB")

                tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

                output = model(tensor)

                # -------------------------------------------------
                # Handle model output
                # -------------------------------------------------
                if isinstance(output, tuple):
                    logits = output[0]
                    features = output[1]
                else:
                    logits = output
                    features = None

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=1,
                )

                predicted_index = int(
                    predicted_index.item()
                )

                confidence = float(
                    confidence.item()
                )

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                y_true.append(true_class)
                y_pred.append(predicted_class)

                results.append({
                    "index": index,
                    "dataset": row["dataset"],
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "path": str(image_path),
                    "sha256": row["sha256"],
                })

            except Exception as exc:

                inference_errors.append({
                    "index": index,
                    "path": str(image_path),
                    "error": repr(exc),
                })

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------
    total = len(y_true)

    if total == 0:
        raise RuntimeError(
            "No images were successfully evaluated."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            average="weighted",
            zero_division=0,
        )
    )

    # -------------------------------------------------------------
    # Per-class metrics
    # -------------------------------------------------------------
    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average=None,
        zero_division=0,
    )

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------
    print()
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(f"Evaluated:       {total}")
    print(f"Inference errors:{len(inference_errors)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")

    # -------------------------------------------------------------
    # Per-class
    # -------------------------------------------------------------
    print()
    print("=" * 80)
    print("PER-CLASS RESULTS")
    print("=" * 80)

    print(
        f"{'Class':25s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'Support':>12s}"
    )

    for i, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:25s}"
            f"{precision[i]:12.4f}"
            f"{recall[i]:12.4f}"
            f"{f1[i]:12.4f}"
            f"{support[i]:12d}"
        )

    # -------------------------------------------------------------
    # Dataset-wise accuracy
    # -------------------------------------------------------------
    result_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("DATASET-WISE RESULTS")
    print("=" * 80)

    for dataset_name, group in result_df.groupby("dataset"):

        dataset_accuracy = (
            group["true_class"]
            == group["predicted_class"]
        ).mean()

        print(
            f"{dataset_name:25s}"
            f"{len(group):6d} images"
            f"  accuracy={dataset_accuracy:.4f}"
        )

    # -------------------------------------------------------------
    # Most common errors
    # -------------------------------------------------------------
    errors_df = result_df[
        result_df["true_class"]
        != result_df["predicted_class"]
    ]

    print()
    print("=" * 80)
    print("MOST COMMON CLASSIFICATION ERRORS")
    print("=" * 80)

    if len(errors_df):

        error_pairs = (
            errors_df
            .groupby(
                ["true_class", "predicted_class"]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        for (true_class, predicted_class), count in (
            error_pairs.head(20).items()
        ):

            print(
                f"{true_class:25s}"
                f" -> "
                f"{predicted_class:25s}"
                f"{count:5d}"
            )

    else:
        print("No classification errors.")

    # -------------------------------------------------------------
    # Save machine-readable results
    # -------------------------------------------------------------
    output_dir = (
        ROOT
        / "runs"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json = (
        output_dir
        / "expanded_heldout_classifier.json"
    )

    report = {
        "evaluation": "expanded_heldout_classifier",
        "manifest": str(CSV_PATH),
        "model": str(MODEL_PATH),
        "device": str(DEVICE),
        "total": total,
        "inference_errors": len(inference_errors),
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "ood_threshold": ood_threshold,
        "classes": CLASS_NAMES,
        "inference_error_details": inference_errors,
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {output_json}")
    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()