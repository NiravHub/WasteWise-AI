"""
WasteWise AI
Expanded ResNet50 - 19-Class OOD Calibration

Builds class centroids from the expanded validation set and
calibrates an OOD threshold using the 1st percentile of
correct-class cosine similarities.

Model:
    models/expanded_resnet50_stage2_best.pth

Validation:
    data/splits_expanded/validation.csv

Output:
    models/expanded_ood_calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.expanded_dataset import ExpandedWasteDataset
from src.models.resnet50_model import WasteResNet50


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "expanded_resnet50_stage2_best.pth"
)

VALIDATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits_expanded"
    / "validation.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "expanded_ood_calibration.json"
)


# ============================================================
# EXACT 19-CLASS ORDER
# ============================================================

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

CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# PREPROCESSING
# ============================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

def load_model(
    device: torch.device,
) -> WasteResNet50:

    print("Loading Expanded ResNet50...")

    model = WasteResNet50(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("WASTEWISE AI - EXPANDED OOD CALIBRATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device:             {device}")
    print(f"Model:              {MODEL_PATH}")
    print(f"Validation CSV:     {VALIDATION_CSV}")
    print(f"Output:             {OUTPUT_PATH}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Validation CSV not found: "
            f"{VALIDATION_CSV}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("Loading validation dataset...")

    dataset = ExpandedWasteDataset(
        csv_path=VALIDATION_CSV,
        transform=TRANSFORM,
        class_to_idx=CLASS_TO_IDX,
    )

    print(
        f"Validation samples: {len(dataset)}"
    )

    print(
        f"Classes:            {len(CLASS_NAMES)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Collect embeddings
    # --------------------------------------------------------

    print()
    print("Extracting 2048-D feature embeddings...")

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (
            images,
            labels,
        ) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True,
            )

            features = model.extract_features(
                images
            )

            features = F.normalize(
                features,
                p=2,
                dim=1,
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            if (
                batch_index + 1
            ) % 20 == 0:

                processed = min(
                    (batch_index + 1) * 32,
                    len(dataset),
                )

                print(
                    f"  processed "
                    f"{processed}/"
                    f"{len(dataset)}"
                )

    features = torch.cat(
        all_features,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    print()
    print(
        f"Feature matrix:     {tuple(features.shape)}"
    )

    # --------------------------------------------------------
    # Build centroids
    # --------------------------------------------------------

    print()
    print("Building class centroids...")

    centroids = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_features = features[
            labels == class_index
        ]

        if len(class_features) == 0:
            raise RuntimeError(
                f"No validation samples found "
                f"for class: {class_name}"
            )

        centroid = class_features.mean(
            dim=0
        )

        centroid = F.normalize(
            centroid,
            p=2,
            dim=0,
        )

        centroids[class_name] = (
            centroid
            .cpu()
            .tolist()
        )

        print(
            f"  {class_name:<20}"
            f"{len(class_features):>5} samples"
        )

    # --------------------------------------------------------
    # Calculate correct-class similarities
    # --------------------------------------------------------

    print()
    print(
        "Calculating validation "
        "correct-class similarities..."
    )

    centroid_matrix = torch.stack(
        [
            torch.tensor(
                centroids[class_name],
                dtype=torch.float32,
            )
            for class_name in CLASS_NAMES
        ]
    )

    centroid_matrix = F.normalize(
        centroid_matrix,
        p=2,
        dim=1,
    )

    similarity_matrix = (
        features
        @ centroid_matrix.T
    )

    correct_similarities = (
        similarity_matrix[
            torch.arange(
                len(labels)
            ),
            labels,
        ]
    )

    similarity_values = (
        correct_similarities
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = float(
        np.percentile(
            similarity_values,
            1,
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_similarity = float(
        np.mean(similarity_values)
    )

    median_similarity = float(
        np.median(similarity_values)
    )

    min_similarity = float(
        np.min(similarity_values)
    )

    max_similarity = float(
        np.max(similarity_values)
    )

    percentile_1 = threshold
    percentile_5 = float(
        np.percentile(
            similarity_values,
            5,
        )
    )

    percentile_10 = float(
        np.percentile(
            similarity_values,
            10,
        )
    )

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    per_class_stats = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_values = similarity_values[
            labels.numpy()
            == class_index
        ]

        per_class_stats[class_name] = {
            "samples": int(
                len(class_values)
            ),
            "mean_similarity": round(
                float(
                    np.mean(
                        class_values
                    )
                ),
                6,
            ),
            "median_similarity": round(
                float(
                    np.median(
                        class_values
                    )
                ),
                6,
            ),
            "min_similarity": round(
                float(
                    np.min(
                        class_values
                    )
                ),
                6,
            ),
            "max_similarity": round(
                float(
                    np.max(
                        class_values
                    )
                ),
                6,
            ),
        }

    # --------------------------------------------------------
    # Save calibration
    # --------------------------------------------------------

    calibration = {
        "method": (
            "Expanded ResNet50 2048D "
            "feature-space cosine similarity"
        ),
        "model": str(
            MODEL_PATH
        ),
        "validation_csv": str(
            VALIDATION_CSV
        ),
        "threshold_method": (
            "1st percentile of "
            "correct-class validation "
            "cosine similarities"
        ),
        "threshold": round(
            threshold,
            6,
        ),
        "feature_dimension": 2048,
        "validation_samples": int(
            len(dataset)
        ),
        "classes": CLASS_NAMES,
        "centroids": centroids,
        "validation_similarity_statistics": {
            "mean": round(
                mean_similarity,
                6,
            ),
            "median": round(
                median_similarity,
                6,
            ),
            "min": round(
                min_similarity,
                6,
            ),
            "max": round(
                max_similarity,
                6,
            ),
            "percentile_1": round(
                percentile_1,
                6,
            ),
            "percentile_5": round(
                percentile_5,
                6,
            ),
            "percentile_10": round(
                percentile_10,
                6,
            ),
        },
        "per_class_statistics": (
            per_class_stats
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            calibration,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OOD CALIBRATION COMPLETE")
    print("=" * 70)

    print(
        f"Validation samples : {len(dataset)}"
    )

    print(
        f"Feature dimension   : 2048"
    )

    print(
        f"Classes             : {len(CLASS_NAMES)}"
    )

    print(
        f"Mean similarity     : "
        f"{mean_similarity:.6f}"
    )

    print(
        f"Median similarity   : "
        f"{median_similarity:.6f}"
    )

    print(
        f"Minimum similarity  : "
        f"{min_similarity:.6f}"
    )

    print(
        f"Maximum similarity  : "
        f"{max_similarity:.6f}"
    )

    print(
        f"1st percentile      : "
        f"{percentile_1:.6f}"
    )

    print(
        f"5th percentile      : "
        f"{percentile_5:.6f}"
    )

    print(
        f"10th percentile     : "
        f"{percentile_10:.6f}"
    )

    print()

    print(
        f"NEW OOD THRESHOLD   : "
        f"{threshold:.6f}"
    )

    print()

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
