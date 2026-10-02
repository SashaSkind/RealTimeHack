"""Save a copy of each image with a labeled pixel grid overlay (every 64 px) to _grid/.

Usage: python grid.py <file.jpg> [...]   (file names relative to photos dir)
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
STEP = 64


def grid(fn):
    im = cv2.imread(str(ROOT / fn))
    h, w = im.shape[:2]
    ov = im.copy()
    for x in range(0, w, STEP):
        major = x % 256 == 0
        cv2.line(ov, (x, 0), (x, h), (0, 255, 255) if major else (0, 200, 0), 2 if major else 1)
    for y in range(0, h, STEP):
        major = y % 256 == 0
        cv2.line(ov, (0, y), (w, y), (0, 255, 255) if major else (0, 200, 0), 2 if major else 1)
    im = cv2.addWeighted(ov, 0.6, im, 0.4, 0)
    for x in range(0, w, STEP):
        for y in range(0, h, STEP * 3):
            cv2.putText(im, str(x), (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(im, str(x), (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    for y in range(0, h, STEP):
        for x in range(0, w, STEP * 4):
            cv2.putText(im, str(y), (x + 2, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(im, str(y), (x + 2, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    out = ROOT / "_grid" / fn
    cv2.imwrite(str(out), im)
    return out


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        print(grid(fn))
