from pathlib import Path

path = Path(r"scripts\build_taco_yolo_dataset_final.py")
text = path.read_text(encoding="utf-8")

old = '''                        # Tiny numerical tolerance
                        eps = 1e-6

                        # Clamp only tiny boundary floating errors.
                        if nx < 0 and nx > -eps:
                            nx = 0

                        if ny < 0 and ny > -eps:
                            ny = 0

                        if nx + nw > actual_W and nx + nw < actual_W + eps:
                            nw = actual_W - nx

                        if ny + nh > actual_H and ny + nh < actual_H + eps:
                            nh = actual_H - ny

                        # Final validity check
                        if (
                            nx < 0
                            or ny < 0
                            or nw <= 0
                            or nh <= 0
                            or nx + nw > actual_W + eps
                            or ny + nh > actual_H + eps
                        ):
                            invalid_boxes += 1
                            continue
'''

new = '''                        # TACO boxes are already expressed in the
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
'''

if old not in text:
    raise SystemExit("Target validation block was not found. No changes made.")

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Patched TACO boundary clipping successfully.")
