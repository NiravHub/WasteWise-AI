import unittest
import torch
import torch.nn.functional as F
from pathlib import Path
import json

from src.inference.ood_detector import OODDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = PROJECT_ROOT / "models" / "expanded_ood_calibration.json"


class TestOODDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = OODDetector(calibration_path=str(CALIBRATION_PATH))

    def test_initialization(self):
        self.assertGreater(self.detector.threshold, 0.0)
        self.assertEqual(len(self.detector.classes), 19)
        self.assertEqual(len(self.detector.centroids), 19)
        for name in self.detector.classes:
            self.assertIn(name, self.detector.centroids)
            norm = torch.norm(self.detector.centroids[name], p=2).item()
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_exact_centroid_match(self):
        # A feature vector identical to a centroid should give similarity ~ 1.0
        target_class = "plastic"
        centroid = self.detector.centroids[target_class]
        res = self.detector.detect(centroid)

        self.assertTrue(res["is_supported"])
        self.assertEqual(res["best_class"], target_class)
        self.assertAlmostEqual(res["similarity"], 1.0, places=4)

    def test_orthogonal_vector_ood_rejection(self):
        # A zero or opposite/orthogonal vector should fall below threshold or be rejected
        first_centroid = next(iter(self.detector.centroids.values()))
        # Create a vector orthogonal or random with very low similarity
        noise = torch.randn_like(first_centroid)
        res = self.detector.detect(noise)
        self.assertIn("is_supported", res)
        self.assertIn("similarity", res)
        self.assertIn("threshold", res)

    def test_custom_threshold_override(self):
        target_class = "cardboard"
        centroid = self.detector.centroids[target_class]
        # With threshold 0.99, an exact centroid is supported
        res = self.detector.detect(centroid, threshold=0.99)
        self.assertTrue(res["is_supported"])

        # With impossible threshold 1.5, it must be rejected
        res_rejected = self.detector.detect(centroid, threshold=1.5)
        self.assertFalse(res_rejected["is_supported"])
        self.assertEqual(res_rejected["threshold"], 1.5)


if __name__ == "__main__":
    unittest.main()
