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

from src.models.resnet50_model import WasteResNet50
from src.data.expanded_transforms import get_eval_transforms
from src.inference.ood_detector import OODDetector


class COCODataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = sorted(
            [
                p for p in Path(image_dir).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )

        self.transform = get_eval_transforms()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, str(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained ResNet50 checkpoint",
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

    image_dir = (
        ROOT
        / "data"
        / "external"
        / "coco"
        / "val2017"
        / "val2017"
    )

    calibration_path = (
        ROOT
        / "models"
        / "ood_calibration.json"
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"COCO image directory not found: {image_dir}"
        )

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"OOD calibration not found: {calibration_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("WasteWise AI - COCO External OOD Evaluation")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    with open(
        calibration_path,
        "r",
        encoding="utf-8",
    ) as f:
        calibration = json.load(f)

    threshold = float(
        calibration["threshold"]
    )

    print(f"OOD threshold: {threshold:.4f}")

    dataset = COCODataset(
        image_dir
    )

    print(
        f"COCO images: {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Build the same 19-class model.
    model = WasteResNet50(
        num_classes=19,
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

    detector = OODDetector(
        calibration_path=str(
            calibration_path
        )
    )

    print()
    print("Running COCO OOD inference...")

    all_max_similarities = []
    all_supported = []
    all_paths = []

    with torch.no_grad():

        for batch_idx, (images, paths) in enumerate(
            loader,
            start=1,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            # Extract the 2048-dimensional ResNet feature.
            features = model.extract_features(
                images
            )

            for feature, path in zip(
                features,
                paths,
            ):

                feature_np = feature.detach().cpu()

                result = detector.detect(
                    feature_np
                )

                all_max_similarities.append(
                    float(
                        result["similarity"]
                    )
                )

                all_supported.append(
                    bool(
                        result["is_supported"]
                    )
                )

                all_paths.append(path)

            if batch_idx % 10 == 0:
                print(
                    f"  Processed "
                    f"{min(batch_idx * 32, len(dataset))}"
                    f"/{len(dataset)}"
                )

    similarities = np.array(
        all_max_similarities
    )

    supported = np.array(
        all_supported,
        dtype=bool,
    )

    rejected = ~supported

    total = len(dataset)

    rejected_count = int(
        rejected.sum()
    )

    accepted_count = int(
        supported.sum()
    )

    rejection_rate = (
        rejected_count / total
        if total
        else 0.0
    )

    false_acceptance_rate = (
        accepted_count / total
        if total
        else 0.0
    )

    metrics = {
        "experiment": (
            f"expanded_resnet50_"
            f"{args.stage}_coco_ood"
        ),
        "dataset": "COCO val2017",
        "num_images": total,
        "ood_threshold": threshold,
        "rejected_as_unsupported": rejected_count,
        "accepted_as_supported": accepted_count,
        "ood_rejection_rate": rejection_rate,
        "false_acceptance_rate": false_acceptance_rate,
        "similarity_min": float(
            similarities.min()
        ),
        "similarity_mean": float(
            similarities.mean()
        ),
        "similarity_median": float(
            np.median(similarities)
        ),
        "similarity_max": float(
            similarities.max()
        ),
        "similarity_std": float(
            similarities.std()
        ),
    }

    output_dir = (
        ROOT
        / "results"
        / "metrics"
        / "expanded"
        / "coco_ood"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / (
            f"expanded_resnet50_"
            f"{args.stage}_coco_ood_metrics.json"
        )
    )

    similarity_path = (
        output_dir
        / (
            f"expanded_resnet50_"
            f"{args.stage}_coco_ood_similarity.csv"
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
        similarity_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "image_path",
                "similarity",
                "threshold",
                "accepted_as_supported",
                "rejected_as_unsupported",
            ]
        )

        for path, similarity, is_supported in zip(
            all_paths,
            all_max_similarities,
            all_supported,
        ):

            writer.writerow(
                [
                    path,
                    similarity,
                    threshold,
                    is_supported,
                    not is_supported,
                ]
            )

    print()
    print("=" * 70)
    print("COCO OOD RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated:       {total}"
    )

    print(
        f"OOD threshold:          {threshold:.4f}"
    )

    print(
        f"Rejected:                {rejected_count}"
    )

    print(
        f"Accepted:                {accepted_count}"
    )

    print(
        f"OOD rejection rate:     {rejection_rate:.4%}"
    )

    print(
        f"False acceptance rate:  {false_acceptance_rate:.4%}"
    )

    print()
    print(
        f"Similarity min:         "
        f"{similarities.min():.4f}"
    )

    print(
        f"Similarity mean:        "
        f"{similarities.mean():.4f}"
    )

    print(
        f"Similarity median:      "
        f"{np.median(similarities):.4f}"
    )

    print(
        f"Similarity max:         "
        f"{similarities.max():.4f}"
    )

    print()
    print("Saved:")
    print(metrics_path)
    print(similarity_path)


if __name__ == "__main__":
    main()

