import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app, set_predictor, get_predictor
from src.inference.predict import WasteWisePredictor


class DummyMockPredictor:
    """Deterministic mock predictor for API unit testing."""
    def __init__(self):
        self.device = "cpu"
        self.detector = True
        self.classifier = True
        self.ood_detector = type("MockOOD", (), {"threshold": 0.25})()
        self.detection_confidence = 0.50

    def predict(self, image_input, confidence_threshold=None, ood_threshold=None):
        return {
            "success": True,
            "version": "1.0.0",
            "duration_ms": 12.34,
            "image": {"width": 100, "height": 100, "filename": "test.jpg"},
            "detector": {
                "model": "mock_yolo.pt",
                "confidence_threshold": confidence_threshold or 0.5,
                "raw_detections": 2,
                "accepted_detections": 1,
                "detections": 2,
            },
            "classifier": {
                "model": "mock_resnet.pth",
                "classes": ["plastic", "glass"],
                "ood_threshold": ood_threshold or 0.25,
            },
            "objects": [
                {
                    "object_id": 1,
                    "bounding_box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50, "width": 40, "height": 40},
                    "detector": {"class": "plastic", "confidence": 95.0},
                    "classifier": {"is_supported": True, "prediction": "plastic", "confidence": 98.0},
                    "accepted": True,
                    "rejection_reason": None,
                }
            ],
            "rejected_objects": [
                {
                    "object_id": 2,
                    "bounding_box": {"x1": 60, "y1": 60, "x2": 90, "y2": 90, "width": 30, "height": 30},
                    "detector": {"class": "plastic", "confidence": 70.0},
                    "classifier": {"is_supported": False, "prediction": None, "confidence": 40.0},
                    "accepted": False,
                    "rejection_reason": "ood_rejection",
                }
            ],
            "summary": {
                "raw_detections": 2,
                "total_objects": 1,
                "accepted_objects": 1,
                "supported_objects": 1,
                "unsupported_objects": 1,
                "ood_rejections": 1,
                "compatibility_rejections": 0,
                "other_rejections": 0,
                "total_rejections": 1,
                "accounting_valid": True,
            },
            "accounting": {
                "raw_detections": 2,
                "accepted_detections": 1,
                "ood_rejected": 1,
                "compatibility_rejected": 0,
                "other_rejected": 0,
                "total_rejected": 1,
                "is_valid": True,
            },
        }


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_predictor(DummyMockPredictor())
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        set_predictor(None)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "WasteWise AI")
        self.assertEqual(data["status"], "running")

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ready"])
        self.assertEqual(data["detector"], "YOLO11s")

    def test_readiness_endpoint(self):
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ready")

    def test_predict_unsupported_extension(self):
        files = {"file": ("test.txt", b"plain text content", "text/plain")}
        resp = self.client.post("/predict", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported image format", resp.json()["detail"])

    def test_predict_empty_file(self):
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        resp = self.client.post("/predict", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())

    def test_predict_corrupt_image(self):
        files = {"file": ("corrupt.jpg", b"not an image binary payload", "image/jpeg")}
        resp = self.client.post("/predict", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Corrupted or invalid image", resp.json()["detail"])

    def test_predict_valid_image(self):
        # Generate valid in-memory JPEG
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        files = {"file": ("sample.jpg", buf.getvalue(), "image/jpeg")}
        resp = self.client.post("/predict", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["filename"], "sample.jpg")
        self.assertIn("duration_ms", data)
        self.assertIn("accounting", data)
        self.assertEqual(data["accounting"]["raw_detections"], 2)
        self.assertEqual(data["accounting"]["accepted_detections"], 1)
        self.assertEqual(len(data["objects"]), 1)
        self.assertEqual(len(data["rejected_objects"]), 1)
        self.assertEqual(data["rejected_objects"][0]["rejection_reason"], "ood_rejection")


if __name__ == "__main__":
    unittest.main()
