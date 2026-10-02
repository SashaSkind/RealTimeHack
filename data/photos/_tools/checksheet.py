"""Tile _check/ images (full view + zoomed crop around main box) for quick verification.

Usage: python checksheet.py <out_name> <file.jpg> [...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = 360


def fit(im, s=T):
    h, w = im.shape[:2]
    k = s / max(h, w)
    im = cv2.resize(im, (max(1, int(w * k)), max(1, int(h * k))))
    out = np.full((s, s, 3), 255, np.uint8)
    out[:im.shape[0], :im.shape[1]] = im
    return out


def main():
    boxes = json.loads((ROOT / "_tools" / "boxes.json").read_text())
    tiles = []
    for fn in sys.argv[2:]:
        im = cv2.imread(str(ROOT / "_check" / fn))
        x1, y1, x2, y2 = boxes[fn]["box"]
        bw, bh = x2 - x1, y2 - y1
        m = int(max(bw, bh) * 0.35) + 10
        crop = im[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        a, b = fit(im), fit(crop)
        pair = np.hstack([a, b])
        cv2.putText(pair, fn, (4, T - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(pair, fn, (4, T - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        tiles.append(pair)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([tiles[i], tiles[i + 1]]) for i in range(0, len(tiles), 2)]
    out = ROOT / "_check" / f"sheet_{sys.argv[1]}.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main()
