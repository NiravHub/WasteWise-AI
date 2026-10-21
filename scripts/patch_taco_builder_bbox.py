from pathlib import Path

path = Path(r"scripts\build_taco_yolo_dataset_final.py")
text = path.read_text(encoding="utf-8")

start = text.index("def transform_bbox(")
end = text.index("\n\n", start)

replacement = '''def transform_bbox(x, y, w, h, W, H, orientation):
    """
    TACO bounding boxes are already expressed in the display-oriented
    coordinate system used by the normalized image.

    The source image is normalized with ImageOps.exif_transpose().
    Therefore, DO NOT apply any additional EXIF rotation to the bbox.

    The coordinates are returned AS-IS.
    """
    return x, y, w, h
'''

text = text[:start] + replacement + text[end:]

path.write_text(text, encoding="utf-8")

print("Patched transform_bbox() successfully.")
