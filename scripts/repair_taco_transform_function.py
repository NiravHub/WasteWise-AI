from pathlib import Path

path = Path(r"scripts\build_taco_yolo_dataset_final.py")
text = path.read_text(encoding="utf-8")

start = text.index("def transform_bbox(")
end = text.index("\ndef read_manifest()", start)

replacement = '''def transform_bbox(x, y, w, h, W, H, orientation):
    """
    TACO bounding boxes are already expressed in the display-oriented
    coordinate system used by the normalized image.

    The source image is normalized with ImageOps.exif_transpose().
    Therefore, no additional EXIF rotation is applied to the bbox.

    Coordinates are returned AS-IS.
    """
    return x, y, w, h

'''

text = text[:start] + replacement + text[end:]

path.write_text(text, encoding="utf-8")

print("transform_bbox() repaired successfully.")
