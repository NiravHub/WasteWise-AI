import json
from pathlib import Path

root = Path("data/external/taco/extracted/TACO/data")

json_files = [root / "annotations.json"] + sorted(root.glob("batch_*/annotations.json"))

print("JSON files:", len(json_files))

for jf in json_files[:3]:
    print("\nFILE:", jf)
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("images:", len(data.get("images", [])))
    print("annotations:", len(data.get("annotations", [])))

    for img in data.get("images", [])[:5]:
        print(img)
