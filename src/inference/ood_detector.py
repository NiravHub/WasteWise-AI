"""
WasteWise AI - Out-of-Distribution Detector

Detects whether an input image is sufficiently similar
to the known waste classes using ResNet50 feature embeddings.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


class OODDetector:
    def __init__(
        self,
        calibration_path="models/ood_calibration.json",
        threshold=None,
    ):
        self.calibration_path = Path(calibration_path)

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                f"OOD calibration file not found: "
                f"{self.calibration_path}"
            )

        with open(self.calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)

        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(calibration["threshold"])
        )
        self.classes = calibration["classes"]

        self.centroids = {}

        for class_name in self.classes:
            centroid = torch.tensor(
                calibration["centroids"][class_name],
                dtype=torch.float32,
            )

            centroid = F.normalize(
                centroid,
                p=2,
                dim=0,
            )

            self.centroids[class_name] = centroid

    def detect(self, feature_vector, threshold=None):
        """
        Compare an input feature vector against
        all known waste-class centroids.

        Returns:
            dict containing:
                is_supported
                best_class
                similarity
                threshold
                similarities
        """

        active_threshold = (
            float(threshold)
            if threshold is not None
            else self.threshold
        )

        if feature_vector.ndim == 1:
            feature_vector = feature_vector.unsqueeze(0)

        feature_vector = F.normalize(
            feature_vector,
            p=2,
            dim=1,
        )

        similarities = {}

        for class_name, centroid in self.centroids.items():

            centroid = centroid.to(feature_vector.device)

            similarity = torch.sum(
                feature_vector * centroid.unsqueeze(0),
                dim=1,
            ).item()

            similarities[class_name] = similarity

        best_class = max(
            similarities,
            key=similarities.get,
        )

        best_similarity = similarities[best_class]

        is_supported = (
            best_similarity >= active_threshold
        )

        return {
            "is_supported": bool(is_supported),
            "best_class": best_class,
            "similarity": float(best_similarity),
            "threshold": float(active_threshold),
            "similarities": {
                key: float(value)
                for key, value in similarities.items()
            },
        }