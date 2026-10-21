from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import random
import shutil

from PIL import Image, ImageOps


PROJECT = Path.cwd()

CLEAN_ANN = PROJECT / "results" / "dataset_audit" / "taco" / "taco_clean_annotations.csv"
MANIFEST = PROJECT / "results" / "dataset_audit" / "taco" / "taco_image_manifest_corrected.csv"

OUTPUT = PROJECT / "data" / "processed" / "taco_detection_corrected"
AUDIT_DIR = PROJECT / "results" / "dataset_audit" / "taco"

SEED = 42

# Verified 23-class TACO subset.
SELECTED_CLASSES = {
    59: "Cigarette",
    58: "Unlabeled litter",
    36: "Plastic film",
    5: "Clear plastic bottle",
    29: "Other plastic",
    39: "Other plastic wrapper",
    12: "Drink can",
    7: "Plastic bottle cap",
    55: "Plastic straw",
    9: "Broken glass",
    57: "Styrofoam piece",
    6: "Glass bottle",
    21: "Disposable plastic cup",
    50: "Pop tab",
    14: "Other carton",
    33: "Normal paper",
    8: "Metal bottle cap",
    27: "Plastic lid",
    20: "Paper cup",
    17: "Corrugated carton",
    0: "Aluminium foil",
    40: "Single-use carrier bag",
    4: "Other plastic bottle",
}

CLASS_IDS = {cat_id: idx for idx, cat_id in enumerate(SELECTED_CLASSES.keys())}
CLASS_NAMES = list(SELECTED_CLASSES.values())


def transform_bbox(x, y, w, h, W, H, orientation):
    """
    Transform a bbox from the TACO coordinate system into the
    physically EXIF-transposed image coordinate system.
    """
    if orientation == 1:
        return x, y, w, h, W, H

    if orientation == 2:
        return W - (x + w), y, w, h, W, H

    if orientation == 3:
        return W - (x + w), H - (y + h), w, h, W, H

    if orientation == 4:
        return x, H - (y + h), w, h, W, H

    if orientation == 5:
        return y, x, h, w, H, W

    if orientation == 6:
        return H - (y + h), x, h, w, H, W

    if orientation == 7:
        return H - (y + h), W - (x + w), h, w, H, W

    if orientation == 8:
        return y, W - (x + w), h, w, H, W

    raise ValueError(f"Unsupported EXIF orientation: {orientation}")


def split_images(image_ids, labels_by_image):
    """
    Deterministic image-level approximately stratified 80/10/10 split.
    Rare classes are handled first so they are distributed as evenly
    as possible across the three splits.
    """
    rng = random.Random(SEED)

    image_ids = list(image_ids)

    class_image_counts = Counter()
    for image_id in image_ids:
        for cls in labels_by_image[image_id]:
            class_image_counts[cls] += 1

    # Target image counts.
    n = len(image_ids)
    n_train = round(n * 0.80)
    n_val = round(n * 0.10)
    n_test = n - n_train - n_val

    capacities = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    targets = {}
    for split, capacity in capacities.items():
        targets[split] = {
            cls: class_image_counts[cls] * capacity / n
            for cls in class_image_counts
        }

    # Rare/multi-label images first.
    rng.shuffle(image_ids)
    image_ids.sort(
        key=lambda image_id: (
            -sum(1.0 / max(class_image_counts[c], 1) for c in labels_by_image[image_id]),
            -len(labels_by_image[image_id]),
        )
    )

    assignments = {
        "train": [],
        "val": [],
        "test": [],
    }

    current = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for image_id in image_ids:
        labels = labels_by_image[image_id]

        available = [
            split
            for split in capacities
            if len(assignments[split]) < capacities[split]
        ]

        best_split = None
        best_score = None

        for split in available:
            score = 0.0

            for cls in labels:
                target = targets[split].get(cls, 0.0)
                current_count = current[split][cls]

                if target > 0:
                    deficit = max(target - current_count, 0.0)
                    score += deficit / target

            remaining = capacities[split] - len(assignments[split])
            score += 0.05 * (remaining / max(capacities[split], 1))

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        assignments[best_split].append(image_id)

        for cls in labels:
            current[best_split][cls] += 1

    for split in assignments:
        assignments[split] = sorted(assignments[split])

    return assignments


def main():
    print("=" * 70)
    print("TACO CORRECTED YOLO DATASET BUILDER")
    print("=" * 70)

    if not CLEAN_ANN.exists():
        raise FileNotFoundError(f"Missing: {CLEAN_ANN}")

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing: {MANIFEST}")

    print(f"\nClean annotations : {CLEAN_ANN}")
    print(f"Corrected manifest: {MANIFEST}")
    print(f"Output            : {OUTPUT}")

    # ------------------------------------------------------------
    # Load corrected image manifest
    # ------------------------------------------------------------
    manifest = {}

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "image_id",
            "file_name",
            "width",
            "height",
            "image_path",
        }

        if not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Manifest missing columns. Found: {reader.fieldnames}"
            )

        for row in reader:
            image_id = int(row["image_id"])

            manifest[image_id] = {
                "file_name": row["file_name"],
                "width": int(float(row["width"])),
                "height": int(float(row["height"])),
                "image_path": Path(row["image_path"]),
            }

    print(f"\nManifest images: {len(manifest)}")

    # ------------------------------------------------------------
    # Load clean annotations
    # ------------------------------------------------------------
    annotations_by_image = defaultdict(list)

    with CLEAN_ANN.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "annotation_id",
            "image_id",
            "category_id",
            "x",
            "y",
            "width",
            "height",
        }

        if not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Annotation CSV missing columns. Found: {reader.fieldnames}"
            )

        for row in reader:
            category_id = int(row["category_id"])

            if category_id not in SELECTED_CLASSES:
                continue

            image_id = int(row["image_id"])

            annotations_by_image[image_id].append({
                "annotation_id": int(row["annotation_id"]),
                "category_id": category_id,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
            })

    selected_image_ids = sorted(annotations_by_image.keys())

    total_boxes = sum(
        len(v) for v in annotations_by_image.values()
    )

    print(f"Selected images : {len(selected_image_ids)}")
    print(f"Selected boxes  : {total_boxes}")

    # Expected values from our previous audit.
    if len(selected_image_ids) != 1388:
        raise RuntimeError(
            f"Expected 1388 selected images, got {len(selected_image_ids)}"
        )

    if total_boxes != 4240:
        raise RuntimeError(
            f"Expected 4240 selected boxes, got {total_boxes}"
        )

    # ------------------------------------------------------------
    # Verify source images
    # ------------------------------------------------------------
    print("\nChecking source images...")

    missing = []

    for image_id in selected_image_ids:
        path = manifest[image_id]["image_path"]

        if not path.exists():
            missing.append((image_id, str(path)))

    if missing:
        print("\nMissing images:")
        for item in missing[:20]:
            print(item)

        raise RuntimeError(
            f"{len(missing)} source images are missing."
        )

    print("Source image check: PASS")

    # ------------------------------------------------------------
    # Build image-level split
    # ------------------------------------------------------------
    labels_by_image = {}

    for image_id, anns in annotations_by_image.items():
        labels_by_image[image_id] = {
            ann["category_id"] for ann in anns
        }

    splits = split_images(
        selected_image_ids,
        labels_by_image
    )

    print("\nSplit:")
    for split, ids in splits.items():
        print(f"  {split:5s}: {len(ids)} images")

    if len(splits["train"]) + len(splits["val"]) + len(splits["test"]) != 1388:
        raise RuntimeError("Split image count mismatch.")

    # ------------------------------------------------------------
    # Recreate output directory
    # ------------------------------------------------------------
    if OUTPUT.exists():
        print("\nRemoving previous corrected output...")
        shutil.rmtree(OUTPUT)

    for split in ("train", "val", "test"):
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Convert images + annotations
    # ------------------------------------------------------------
    print("\nConverting images and bounding boxes...")

    split_box_counts = Counter()
    class_box_counts = Counter()
    class_image_counts = Counter()

    dimension_mismatches = []
    invalid_boxes = []
    converted_boxes = 0

    split_lookup = {}

    for split, ids in splits.items():
        for image_id in ids:
            split_lookup[image_id] = split

    for index, image_id in enumerate(selected_image_ids, start=1):
        split = split_lookup[image_id]
        info = manifest[image_id]

        source_path = info["image_path"]
        expected_W = info["width"]
        expected_H = info["height"]

        with Image.open(source_path) as original:
            orientation = int(original.getexif().get(274, 1) or 1)

            normalized = ImageOps.exif_transpose(original).convert("RGB")

            out_W, out_H = normalized.size

            if (out_W, out_H) != (expected_W, expected_H):
                dimension_mismatches.append({
                    "image_id": image_id,
                    "file_name": info["file_name"],
                    "expected": [expected_W, expected_H],
                    "actual": [out_W, out_H],
                    "orientation": orientation,
                })

                raise RuntimeError(
                    f"Dimension mismatch after EXIF normalization: "
                    f"{info['file_name']} "
                    f"expected={expected_W}x{expected_H}, "
                    f"actual={out_W}x{out_H}, "
                    f"orientation={orientation}"
                )

            output_name = f"taco_{image_id:04d}.jpg"

            output_image = (
                OUTPUT / split / "images" / output_name
            )

            output_label = (
                OUTPUT / split / "labels" /
                f"taco_{image_id:04d}.txt"
            )

            normalized.save(
                output_image,
                "JPEG",
                quality=95,
                optimize=True
            )

            yolo_lines = []

            for ann in annotations_by_image[image_id]:
                x = ann["x"]
                y = ann["y"]
                w = ann["width"]
                h = ann["height"]

                tx, ty, tw, th, final_W, final_H = transform_bbox(
                    x,
                    y,
                    w,
                    h,
                    expected_W,
                    expected_H,
                    orientation
                )

                # Small floating-point tolerance only.
                eps = 1e-5

                if tx < -eps or ty < -eps:
                    invalid_boxes.append({
                        "image_id": image_id,
                        "annotation_id": ann["annotation_id"],
                        "reason": "negative coordinate",
                        "bbox": [tx, ty, tw, th],
                    })
                    continue

                if tx + tw > final_W + eps or ty + th > final_H + eps:
                    invalid_boxes.append({
                        "image_id": image_id,
                        "annotation_id": ann["annotation_id"],
                        "reason": "out of bounds",
                        "bbox": [tx, ty, tw, th],
                        "image_size": [final_W, final_H],
                    })
                    continue

                if tw <= 0 or th <= 0:
                    invalid_boxes.append({
                        "image_id": image_id,
                        "annotation_id": ann["annotation_id"],
                        "reason": "non-positive size",
                        "bbox": [tx, ty, tw, th],
                    })
                    continue

                # Clamp tiny floating-point drift.
                tx = max(0.0, min(float(tx), float(final_W)))
                ty = max(0.0, min(float(ty), float(final_H)))
                tw = min(float(tw), float(final_W - tx))
                th = min(float(th), float(final_H - ty))

                xc = (tx + tw / 2.0) / final_W
                yc = (ty + th / 2.0) / final_H
                nw = tw / final_W
                nh = th / final_H

                class_id = CLASS_IDS[ann["category_id"]]

                yolo_lines.append(
                    f"{class_id} "
                    f"{xc:.8f} "
                    f"{yc:.8f} "
                    f"{nw:.8f} "
                    f"{nh:.8f}"
                )

                converted_boxes += 1
                split_box_counts[split] += 1
                class_box_counts[ann["category_id"]] += 1

            if not yolo_lines:
                raise RuntimeError(
                    f"No valid boxes generated for image {image_id}"
                )

            output_label.write_text(
                "\n".join(yolo_lines) + "\n",
                encoding="utf-8"
            )

            for cls in labels_by_image[image_id]:
                class_image_counts[cls] += 1

        if index % 100 == 0 or index == len(selected_image_ids):
            print(
                f"  Processed {index}/{len(selected_image_ids)}"
            )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------
    print("\nFinal validation...")

    image_count = 0
    label_count = 0

    for split in ("train", "val", "test"):
        image_files = sorted(
            (OUTPUT / split / "images").glob("*.jpg")
        )

        label_files = sorted(
            (OUTPUT / split / "labels").glob("*.txt")
        )

        image_count += len(image_files)
        label_count += len(label_files)

        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}

        if image_stems != label_stems:
            raise RuntimeError(
                f"Image/label mismatch in split: {split}"
            )

    if image_count != 1388:
        raise RuntimeError(
            f"Final image count mismatch: {image_count}"
        )

    if label_count != 1388:
        raise RuntimeError(
            f"Final label-file count mismatch: {label_count}"
        )

    if converted_boxes != 4240:
        raise RuntimeError(
            f"Converted box count mismatch: {converted_boxes}"
        )

    if invalid_boxes:
        raise RuntimeError(
            f"Invalid boxes encountered: {len(invalid_boxes)}"
        )

    # ------------------------------------------------------------
    # data.yaml
    # ------------------------------------------------------------
    yaml_lines = [
        "path: data/processed/taco_detection_corrected",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]

    for class_id, name in enumerate(CLASS_NAMES):
        yaml_lines.append(
            f"  {class_id}: {name}"
        )

    (OUTPUT / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Class map
    # ------------------------------------------------------------
    class_map_path = OUTPUT / "class_map.csv"

    with class_map_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "yolo_class_id",
            "taco_category_id",
            "class_name",
            "box_count",
            "image_count",
        ])

        for taco_id, name in SELECTED_CLASSES.items():
            writer.writerow([
                CLASS_IDS[taco_id],
                taco_id,
                name,
                class_box_counts[taco_id],
                class_image_counts[taco_id],
            ])

    # ------------------------------------------------------------
    # Split manifest
    # ------------------------------------------------------------
    split_manifest_path = (
        AUDIT_DIR / "taco_yolo_split_manifest.csv"
    )

    with split_manifest_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "split",
            "file_name",
            "image_path",
            "width",
            "height",
            "orientation",
            "annotation_count",
        ])

        for split, ids in splits.items():
            for image_id in ids:
                info = manifest[image_id]

                with Image.open(info["image_path"]) as im:
                    orientation = int(
                        im.getexif().get(274, 1) or 1
                    )

                writer.writerow([
                    image_id,
                    split,
                    info["file_name"],
                    str(info["image_path"]),
                    info["width"],
                    info["height"],
                    orientation,
                    len(annotations_by_image[image_id]),
                ])

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    summary = {
        "dataset": "TACO",
        "purpose": "YOLO object detection",
        "selected_classes": len(SELECTED_CLASSES),
        "selected_images": len(selected_image_ids),
        "selected_annotations": total_boxes,
        "converted_boxes": converted_boxes,
        "invalid_boxes": len(invalid_boxes),
        "dimension_mismatches": len(dimension_mismatches),
        "split": {
            split: {
                "images": len(splits[split]),
                "boxes": split_box_counts[split],
            }
            for split in ("train", "val", "test")
        },
        "seed": SEED,
        "exif_normalization": True,
        "source_manifest": str(MANIFEST),
        "source_annotations": str(CLEAN_ANN),
        "output": str(OUTPUT),
        "status": "PASS",
    }

    summary_path = (
        AUDIT_DIR / "taco_yolo_build_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print(f"\nImages          : {image_count}")
    print(f"Label files     : {label_count}")
    print(f"Converted boxes : {converted_boxes}")
    print(f"Invalid boxes   : {len(invalid_boxes)}")
    print(f"Dim mismatches  : {len(dimension_mismatches)}")

    print("\nSplits:")
    for split in ("train", "val", "test"):
        print(
            f"  {split:5s}: "
            f"{len(splits[split])} images, "
            f"{split_box_counts[split]} boxes"
        )

    print("\nClass distribution:")
    for taco_id, name in SELECTED_CLASSES.items():
        print(
            f"  {CLASS_IDS[taco_id]:2d} "
            f"{name:28s} "
            f"{class_box_counts[taco_id]:4d} boxes / "
            f"{class_image_counts[taco_id]:4d} images"
        )

    print("\nCreated:")
    print(f"  {OUTPUT}")
    print(f"  {OUTPUT / 'data.yaml'}")
    print(f"  {class_map_path}")
    print(f"  {split_manifest_path}")
    print(f"  {summary_path}")

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
