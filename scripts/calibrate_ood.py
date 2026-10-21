"""
WasteWise AI - OOD Calibration

Calibrates a simple feature-space rejection threshold using
the validation set.

The validation set contains only supported waste classes.
We measure how similar validation images are to the feature
centroid of their predicted class.

The resulting threshold is saved to:
models/ood_calibration.json
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.resnet50_model import WasteResNet50


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/resnet50_stage2_best.pth"
)

OUTPUT_PATH = Path(
    "models/ood_calibration.json"
)

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("WASTEWISE AI - OOD CALIBRATION")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        _,
        validation_loader,
        _,
        class_names,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    print(
        f"Validation images: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Classes: {len(class_names)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = WasteResNet50(
        num_classes=len(class_names),
        pretrained=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Collect feature vectors
    # --------------------------------------------------------

    all_features = []
    all_labels = []

    print("\nExtracting validation features...")

    with torch.no_grad():

        for images, labels in tqdm(
            validation_loader
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            features = model.extract_features(
                images
            )

            # Normalize feature vectors
            features = F.normalize(
                features,
                p=2,
                dim=1
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels
            )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    print(
        f"\nFeature matrix: {features.shape}"
    )

    # --------------------------------------------------------
    # Calculate class centroids
    # --------------------------------------------------------

    centroids = {}

    for class_index, class_name in enumerate(
        class_names
    ):

        class_features = features[
            labels == class_index
        ]

        centroid = class_features.mean(
            dim=0
        )

        centroid = F.normalize(
            centroid,
            p=2,
            dim=0
        )

        centroids[class_name] = centroid

    # --------------------------------------------------------
    # Calculate similarity to correct class centroid
    # --------------------------------------------------------

    similarities = []

    for feature, label in zip(
        features,
        labels
    ):

        class_name = class_names[
            label.item()
        ]

        centroid = centroids[
            class_name
        ]

        similarity = torch.dot(
            feature,
            centroid
        ).item()

        similarities.append(
            similarity
        )

    similarities_tensor = torch.tensor(
        similarities
    )

    # --------------------------------------------------------
    # Calculate conservative threshold
    # --------------------------------------------------------

    # We want almost all known validation images to
    # remain accepted.

    threshold = torch.quantile(
        similarities_tensor,
        0.01
    ).item()

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Convert class centroids to JSON-safe lists
    # --------------------------------------------------------

    centroid_data = {}

    for class_name, centroid in centroids.items():

        centroid_data[class_name] = [
            round(float(value), 8)
            for value in centroid.tolist()
        ]

    class_statistics = {}

    for class_index, class_name in enumerate(
        class_names
    ):

        class_values = similarities_tensor[
            labels == class_index
        ]

        class_statistics[class_name] = {
            "count": int(
                len(class_values)
            ),
            "mean_similarity": round(
                class_values.mean().item(),
                4
            ),
            "min_similarity": round(
                class_values.min().item(),
                4
            ),
            "p01_similarity": round(
                torch.quantile(
                    class_values,
                    0.01
                ).item(),
                4
            ),
            "p05_similarity": round(
                torch.quantile(
                    class_values,
                    0.05
                ).item(),
                4
            ),
        }

    # --------------------------------------------------------
    # Save calibration
    # --------------------------------------------------------

        calibration = {

        "method": (
            "ResNet50 2048D feature-space "
            "cosine similarity"
        ),

        "threshold": round(
            threshold,
            4
        ),

        "feature_dimension": int(
            features.shape[1]
        ),

        "validation_samples": int(
            len(features)
        ),

        "classes": class_names,

        "centroids": centroid_data,

        "class_statistics": (
            class_statistics
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            calibration,
            indent=4
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OOD CALIBRATION COMPLETE")
    print("=" * 70)

    print(
        f"\nThreshold          : "
        f"{threshold:.4f}"
    )

    print(
        f"Minimum similarity : "
        f"{similarities_tensor.min().item():.4f}"
    )

    print(
        f"Mean similarity    : "
        f"{similarities_tensor.mean().item():.4f}"
    )

    print(
        f"Maximum similarity : "
        f"{similarities_tensor.max().item():.4f}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()