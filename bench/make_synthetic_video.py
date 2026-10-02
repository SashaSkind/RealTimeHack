"""Test D input: the Test B photos as a 24 s, 30 fps, 960x720 video. Each photo is shown for
0.5 s (15 frames) with a slow pan/zoom plus a little hand-held jitter, so the tracker has
real motion to follow instead of identical frames."""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS, PER = 960, 720, 30, 15
rows = list(csv.DictReader(open(ROOT / "data/photos/labels.csv")))
out = ROOT / "data/clips/synthetic_photos.mp4"
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rng = np.random.default_rng(0)
for r in rows:
    img = cv2.imread(str(ROOT / "data/photos" / r["file"]))
    h, w = img.shape[:2]
    s = max(W / w, H / h) * 1.15                  # cover the frame with some margin to pan in
    base = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    # centre the crop on the labelled object so it stays in view
    cx = (float(r["x1"]) + float(r["x2"])) / 2 * s
    cy = (float(r["y1"]) + float(r["y2"])) / 2 * s
    x0 = np.clip(cx - W / 2, 0, bw - W)
    y0 = np.clip(cy - H / 2, 0, bh - H)
    vx, vy = rng.uniform(-2, 2, 2)
    for k in range(PER):
        jx, jy = rng.normal(0, 1.5, 2)
        x = int(np.clip(x0 + vx * k + jx, 0, bw - W))
        y = int(np.clip(y0 + vy * k + jy, 0, bh - H))
        vw.write(base[y:y + H, x:x + W])
vw.release()
print(out, len(rows) * PER, "frames")
