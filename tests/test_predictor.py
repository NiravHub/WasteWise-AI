import io
import unittest
from pathlib import Path
from PIL import Image
import torch

from src.inference.predict import WasteWisePredictor
from src.models.resnet50_model import WasteResNet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "external" / "coco" / "val2017" / "val2017" / "000000000139.jpg"


class TestWasteWisePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = WasteWisePredictor(detection_confidence=0.30)

    def test_predictor_initialization(self):
        self.assertIsNotNone(self.predictor.detector)
        self.assertIsNotNone(self.predictor.classifier)
        self.assertIsNotNone(self.predictor.ood_detector)
        self.assertEqual(len(self.predictor.CLASSIFIER_CLASSES), 19)
        self.assertEqual(len(self.predictor.DETECTOR_CLASSES), 6)

    def test_single_image_inference_schema(self):
        if not SAMPLE_IMAGE.exists():
            self.skipTest(f"Sample image not found: {SAMPLE_IMAGE}")

        result = self.predictor.predict(str(SAMPLE_IMAGE))

        self.assertTrue(result["success"])
        self.assertIn("version", result)
        self.assertIn("duration_ms", result)
        self.assertGreater(result["duration_ms"], 0.0)
        self.assertIn("image", result)
        self.assertIn("detector", result)
        self.assertIn("classifier", result)
        self.assertIn("objects", result)
        self.assertIn("rejected_objects", result)
        self.assertIn("summary", result)
        self.assertIn("accounting", result)

    def test_detection_accounting_invariant(self):
        if not SAMPLE_IMAGE.exists():
            self.skipTest(f"Sample image not found: {SAMPLE_IMAGE}")

        result = self.predictor.predict(str(SAMPLE_IMAGE))
        accounting = result["accounting"]

        raw = accounting["raw_detections"]
        accepted = accounting["accepted_detections"]
        ood_rej = accounting["ood_rejected"]
        compat_rej = accounting["compatibility_rejected"]
        other_rej = accounting["other_rejected"]
        total_rej = accounting["total_rejected"]

        # Exact invariant
        self.assertEqual(total_rej, ood_rej + compat_rej + other_rej)
        self.assertEqual(raw, accepted + total_rej)
        self.assertTrue(accounting["is_valid"])
        self.assertEqual(len(result["objects"]), accepted)
        self.assertEqual(len(result["rejected_objects"]), total_rej)

    def test_pil_image_and_stream_inputs(self):
        if not SAMPLE_IMAGE.exists():
            self.skipTest(f"Sample image not found: {SAMPLE_IMAGE}")

        # Test PIL image
        with Image.open(SAMPLE_IMAGE) as img:
            res_pil = self.predictor.predict(img)
            self.assertTrue(res_pil["success"])

        # Test in-memory BytesIO
        with open(SAMPLE_IMAGE, "rb") as f:
            bytes_stream = io.BytesIO(f.read())
            res_stream = self.predictor.predict(bytes_stream)
            self.assertTrue(res_stream["success"])

    def test_empty_blank_image(self):
        # A blank solid-color image should produce 0 raw detections or valid accounting
        blank = Image.new("RGB", (640, 640), color=(128, 128, 128))
        result = self.predictor.predict(blank, confidence_threshold=0.80)
        self.assertTrue(result["success"])
        self.assertTrue(result["accounting"]["is_valid"])
        self.assertEqual(result["accounting"]["raw_detections"], 0)
        self.assertEqual(len(result["objects"]), 0)
        self.assertEqual(len(result["rejected_objects"]), 0)

    def test_invalid_crop_handling(self):
        # Empty or degenerate crop should return safe invalid dictionary without throwing
        invalid_crop = Image.new("RGB", (0, 0))
        res = self.predictor.classify_crop(invalid_crop)
        self.assertFalse(res["is_supported"])
        self.assertEqual(res["category"], "Invalid Crop")

    def test_head_fc_forward_equivalence(self):
        # Verify that evaluating classifier head via backbone.fc matches full model(x)
        model = self.predictor.classifier
        x = torch.randn(1, 3, 224, 224, device=self.predictor.device)
        with torch.no_grad():
            feat = model.extract_features(x)
            logits_full = model(x)
            logits_fc = model.backbone.fc(feat)
        diff = (logits_full - logits_fc).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_compatibility_rules(self):
        compat = self.predictor.DETECTOR_CLASS_COMPATIBILITY
        # Plastic supports plastic, medical_cap, etc., but NOT biological or clothes
        self.assertIn("plastic", compat["plastic"])
        self.assertNotIn("biological", compat["plastic"])
        self.assertNotIn("clothes", compat["plastic"])

        # Paper/cardboard supports paper and cardboard, but NOT electronic_device
        self.assertIn("cardboard", compat["paper_cardboard"])
        self.assertIn("paper", compat["paper_cardboard"])
        self.assertNotIn("electronic_device", compat["paper_cardboard"])

        # Other trash supports general_trash, clothes, shoes
        self.assertIn("general_trash", compat["other_trash"])
        self.assertIn("clothes", compat["other_trash"])
        self.assertIn("shoes", compat["other_trash"])

    def test_battery_rejection_on_coarse_paper_detector(self):
        # Regression test for E4 battery image: YOLO predicts paper_cardboard, ResNet predicts battery
        compat = self.predictor.DETECTOR_CLASS_COMPATIBILITY
        self.assertNotIn("battery", compat["paper_cardboard"])
        # Metal detector on the other hand correctly includes battery
        self.assertIn("battery", compat["metal"])


if __name__ == "__main__":
    unittest.main()
