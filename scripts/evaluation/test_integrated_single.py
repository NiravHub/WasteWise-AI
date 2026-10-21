from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import WasteWisePredictor


image_dir = (
    ROOT
    / "data"
    / "external"
    / "coco"
    / "val2017"
    / "val2017"
)

image_path = next(image_dir.glob("*.jpg"))

print("=" * 70)
print("WASTEWISE INTEGRATED SINGLE-IMAGE TEST")
print("=" * 70)
print(f"Image: {image_path}")

predictor = WasteWisePredictor()

result = predictor.predict(
    str(image_path)
)

print()
print("RESULT:")
print(result)