from pathlib import Path

p = Path(r"scripts\build_taco_yolo_dataset_final.py")

text = p.read_text(encoding="utf-8")

old = """    if orientation == 6:
        nx = H - (y + h)
        ny = x
        nw = h
        nh = w

    elif orientation == 8:
        nx = y
        ny = W - (x + w)
        nw = h
        nh = w

    else:
        nx = x
        ny = y
        nw = w
        nh = h
"""

new = """    if orientation == 3:
        # 180-degree rotation
        nx = W - (x + w)
        ny = H - (y + h)
        nw = w
        nh = h

    elif orientation == 6:
        # 90-degree clockwise rotation
        nx = W - (y + h)
        ny = x
        nw = h
        nh = w

    elif orientation == 8:
        # 90-degree counter-clockwise rotation
        nx = y
        ny = H - (x + w)
        nw = h
        nh = w

    else:
        nx = x
        ny = y
        nw = w
        nh = h
"""

if old not in text:
    raise RuntimeError("Original transformation block was not found.")

p.write_text(text.replace(old, new), encoding="utf-8")

print("SUCCESS: TACO orientation transformation corrected.")
