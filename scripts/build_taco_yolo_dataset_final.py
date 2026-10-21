from pathlib import Path
import json
import csv
import random
import shutil
from collections import defaultdict, Counter

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]

MASTER_JSON = ROOT / "data" / "external" / "taco" / "extracted" / "TACO" / "data" / "annotations.json"
MANIFEST = ROOT / "results" / "dataset_audit" / "taco" / "taco_image_manifest_corrected.csv"

OUT = ROOT / "data" / "processed" / "taco_detection"

SEED = 42

# Final 23-class detector
SELECTED = [
    (59, "Cigarette"),
    (58, "Unlabeled litter"),
    (36, "Plastic film"),
    (5, "Clear plastic bottle"),
    (29, "Other plastic"),
    (39, "Other plastic wrapper"),
    (12, "Drink can"),
    (7, "Plastic bottle cap"),
    (55, "Plastic straw"),
    (9, "Broken glass"),
    (57, "Styrofoam piece"),
    (6, "Glass bottle"),
    (21, "Disposable plastic cup"),
    (50, "Pop tab"),
    (14, "Other carton"),
    (33, "Normal paper"),
    (8, "Metal bottle cap"),
    (27, "Plastic lid"),
    (20, "Paper cup"),
    (17, "Corrugated carton"),
    (0, "Aluminium foil"),
    (40, "Single-use carrier bag"),
    (4, "Other plastic bottle"),
]

CAT_TO_YOLO = {
    taco_id: idx for idx, (taco_id, _) in enumerate(SELECTED)
}

CAT_NAMES = {
    idx: name for idx, (_, name) in enumerate(SELECTED)
}


def transform_bbox(x, y, w, h, W, H, orientation):
    """
    TACO bounding boxes are already expressed in the display-oriented
    coordinate system used by the normalized image.

    The source image is normalized with ImageOps.exif_transpose().
    Therefore, no additional EXIF rotation is applied to the bbox.

    Coordinates are returned AS-IS.
    """
    return x, y, w, h


def read_manifest():
    rows = {}

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = int(row["image_id"])

            rows[image_id] = {
                "file_name": row["file_name"],
                "image_path": Path(row["image_path"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }

    return rows


def split_images(image_ids):
    """
    Deterministic image-level split.

    80% train
    10% val
    10% test
    """

    ids = list(image_ids)

    rng = random.Random(SEED)
    rng.shuffle(ids)

    n = len(ids)

    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def main():

    print("=" * 70)
    print("FINAL TACO → YOLO DATASET BUILDER")
    print("=" * 70)

    with MASTER_JSON.open("r", encoding="utf-8") as f:
        taco = json.load(f)

    manifest = read_manifest()

    print(f"Master images: {len(taco['images'])}")
    print(f"Manifest images: {len(manifest)}")

    # ------------------------------------------------------------
    # Index annotations
    # ------------------------------------------------------------

    annotations_by_image = defaultdict(list)

    selected_annotation_count = 0

    for ann in taco["annotations"]:

        category_id = int(ann["category_id"])

        if category_id not in CAT_TO_YOLO:
            continue

        image_id = int(ann["image_id"])

        annotations_by_image[image_id].append(ann)

        selected_annotation_count += 1

    selected_images = sorted(annotations_by_image.keys())

    print(f"Selected images: {len(selected_images)}")
    print(f"Selected annotations: {selected_annotation_count}")

    if len(selected_images) != 1388:
        print(
            f"WARNING: expected 1388 selected images, "
            f"got {len(selected_images)}"
        )

    if selected_annotation_count != 4240:
        print(
            f"WARNING: expected 4240 selected annotations, "
            f"got {selected_annotation_count}"
        )

    # ------------------------------------------------------------
    # Split at IMAGE level
    # ------------------------------------------------------------

    splits = split_images(selected_images)

    print()
    print("IMAGE SPLIT")
    print("-" * 40)

    for split_name, ids in splits.items():
        print(f"{split_name:5s}: {len(ids)} images")

    # ------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------

    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Process images + boxes
    # ------------------------------------------------------------

    total_boxes = 0
    skipped_boxes = 0
    invalid_boxes = 0
    dimension_mismatches = 0
    orientation_counts = Counter()

    split_annotation_counts = Counter()
    split_class_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    processed_images = 0

    for split_name, image_ids in splits.items():

        print()
        print(f"Processing {split_name}...")

        for image_id in image_ids:

            if image_id not in manifest:
                print(f"ERROR: missing manifest entry for image {image_id}")
                continue

            info = manifest[image_id]
            source_path = info["image_path"]

            if not source_path.exists():
                print(f"ERROR: missing image: {source_path}")
                continue

            try:
                with Image.open(source_path) as original:

                    orientation = int(
                        original.getexif().get(274, 1)
                    )

                    orientation_counts[orientation] += 1

                    # IMPORTANT:
                    # normalize EXIF orientation physically
                    image = ImageOps.exif_transpose(original)

                    actual_W, actual_H = image.size

                    expected_W = info["width"]
                    expected_H = info["height"]

                    if (actual_W, actual_H) != (expected_W, expected_H):
                        dimension_mismatches += 1

                        print(
                            f"DIMENSION ERROR image {image_id}: "
                            f"expected {expected_W}x{expected_H}, "
                            f"got {actual_W}x{actual_H}"
                        )

                        continue

                    # Force RGB
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    output_name = f"taco_{image_id:04d}.jpg"

                    image_out = OUT / "images" / split_name / output_name
                    label_out = OUT / "labels" / split_name / (
                        f"taco_{image_id:04d}.txt"
                    )

                    image.save(
                        image_out,
                        quality=95,
                        subsampling=0
                    )

                    label_lines = []

                    for ann in annotations_by_image[image_id]:

                        taco_cat = int(ann["category_id"])

                        x = float(ann["bbox"][0])
                        y = float(ann["bbox"][1])
                        w = float(ann["bbox"][2])
                        h = float(ann["bbox"][3])

                        if w <= 0 or h <= 0:
                            invalid_boxes += 1
                            continue

                        nx, ny, nw, nh = transform_bbox(
                            x,
                            y,
                            w,
                            h,
                            expected_W,
                            expected_H,
                            orientation
                        )

                        # TACO boxes are already expressed in the
                        # display-oriented coordinate system.
                        #
                        # Clip boxes to the normalized image boundary.
                        # The AS-IS diagnostic found only 6 annotations
                        # with tiny negative coordinates, all in normal
                        # orientation. We preserve those annotations rather
                        # than discarding them.

                        original_nx = nx
                        original_ny = ny
                        original_nw = nw
                        original_nh = nh

                        left = max(0.0, nx)
                        top = max(0.0, ny)
                        right = min(actual_W, nx + nw)
                        bottom = min(actual_H, ny + nh)

                        nx = left
                        ny = top
                        nw = right - left
                        nh = bottom - top

                        # Final validity check
                        if (
                            nx < 0
                            or ny < 0
                            or nw <= 0
                            or nh <= 0
                            or nx + nw > actual_W
                            or ny + nh > actual_H
                        ):
                            invalid_boxes += 1
                            continue

                        xc = (nx + nw / 2) / actual_W
                        yc = (ny + nh / 2) / actual_H
                        bw = nw / actual_W
                        bh = nh / actual_H

                        # YOLO safety check
                        if not (
                            0 <= xc <= 1
                            and 0 <= yc <= 1
                            and 0 < bw <= 1
                            and 0 < bh <= 1
                        ):
                            invalid_boxes += 1
                            continue

                        yolo_class = CAT_TO_YOLO[taco_cat]

                        label_lines.append(
                            f"{yolo_class} "
                            f"{xc:.8f} "
                            f"{yc:.8f} "
                            f"{bw:.8f} "
                            f"{bh:.8f}"
                        )

                        total_boxes += 1
                        split_annotation_counts[split_name] += 1
                        split_class_counts[split_name][yolo_class] += 1

                    if not label_lines:
                        # Should not happen because image was selected
                        skipped_boxes += len(annotations_by_image[image_id])
                        image_out.unlink(missing_ok=True)
                        continue

                    label_out.write_text(
                        "\n".join(label_lines) + "\n",
                        encoding="utf-8"
                    )

                    processed_images += 1

            except Exception as e:
                print(
                    f"ERROR processing image {image_id}: "
                    f"{type(e).__name__}: {e}"
                )

    # ------------------------------------------------------------
    # Dataset YAML
    # ------------------------------------------------------------

    yaml_lines = [
        f"path: {OUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CAT_NAMES)}",
        "names:",
    ]

    for idx in range(len(CAT_NAMES)):
        yaml_lines.append(
            f"  {idx}: {CAT_NAMES[idx]}"
        )

    (OUT / "taco.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Class map
    # ------------------------------------------------------------

    with (OUT / "class_mapping.csv").open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "yolo_id",
            "taco_category_id",
            "class_name"
        ])

        for yolo_id, (taco_id, name) in enumerate(SELECTED):
            writer.writerow([
                yolo_id,
                taco_id,
                name
            ])

    # ------------------------------------------------------------
    # Split manifest
    # ------------------------------------------------------------

    with (OUT / "split_manifest.csv").open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "split",
            "source_file"
        ])

        for split_name, ids in splits.items():
            for image_id in ids:
                writer.writerow([
                    image_id,
                    split_name,
                    manifest[image_id]["file_name"]
                ])

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "images_selected": len(selected_images),
        "images_processed": processed_images,
        "annotations_selected": selected_annotation_count,
        "annotations_converted": total_boxes,
        "skipped_boxes": skipped_boxes,
        "invalid_boxes": invalid_boxes,
        "dimension_mismatches": dimension_mismatches,
        "num_classes": len(SELECTED),
        "seed": SEED,
        "splits": {
            split: len(ids)
            for split, ids in splits.items()
        },
        "orientation_counts": dict(orientation_counts),
        "class_names": CAT_NAMES,
    }

    (OUT / "build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("FINAL BUILD RESULT")
    print("=" * 70)

    print(f"Selected images:       {len(selected_images)}")
    print(f"Processed images:      {processed_images}")
    print(f"Selected annotations:  {selected_annotation_count}")
    print(f"Converted boxes:       {total_boxes}")
    print(f"Skipped boxes:         {skipped_boxes}")
    print(f"Invalid boxes:         {invalid_boxes}")
    print(f"Dimension mismatches:  {dimension_mismatches}")

    print()
    print("SPLITS")
    for split_name in ["train", "val", "test"]:
        print(
            f"{split_name:5s}: "
            f"{len(splits[split_name])} images, "
            f"{split_annotation_counts[split_name]} boxes"
        )

    print()
    print("EXIF orientations:")
    for orientation, count in sorted(orientation_counts.items()):
        print(f"  {orientation}: {count}")

    print()
    print(f"Output: {OUT}")

    if (
        processed_images == 1388
        and total_boxes == 4240
        and skipped_boxes == 0
        and invalid_boxes == 0
        and dimension_mismatches == 0
    ):
        print()
        print("PASS: FINAL TACO YOLO DATASET IS CLEAN")
    else:
        print()
        print("WARNING: DATASET DID NOT MATCH EXPECTED FINAL COUNTS")


if __name__ == "__main__":
    main()
