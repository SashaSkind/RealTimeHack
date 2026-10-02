"""Convert a Stray Scanner recording (free iPhone LiDAR app) so `--source video` replays it in real
time WITH LiDAR depth.

Stray Scanner folder (export via the iOS share menu / Files):
  rgb.mp4              colour video
  depth/000000.png     uint16 depth in millimetres, 256x192 (one per recorded frame)
  confidence/000000.png  0/1/2 LiDAR confidence (optional)

Writes <folder>/rgb.mp4.depth.npz (float16 metres, low-confidence pixels = 0), which
handguide.sources.VideoSource picks up automatically. If the depth count differs from the video
frame count, depth frames are matched to video frames by relative position.

  python bench/stray_to_sidecar.py data/recordings/reach_cup_left
  python -m handguide.main --source video --path data/recordings/reach_cup_left/rgb.mp4 --target cup --display
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def convert(folder: Path, min_conf: int = 1) -> Path:
    folder = Path(folder)
    video = folder / "rgb.mp4"
    dfiles = sorted((folder / "depth").glob("*.png"))
    if not video.exists() or not dfiles:
        raise SystemExit(f"expected {video} and {folder / 'depth'}/*.png (Stray Scanner export)")
    cap = cv2.VideoCapture(str(video))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if n_video <= 0:
        raise SystemExit(f"could not read frame count of {video}")

    cdir = folder / "confidence"
    depths = []
    for f in dfiles:
        d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            raise SystemExit(f"{f}: expected a 16-bit PNG (millimetres)")
        m = d.astype(np.float32) / 1000.0
        cf = cdir / f.name
        if cf.exists():
            c = cv2.imread(str(cf), cv2.IMREAD_UNCHANGED)
            if c is not None and c.shape == m.shape:
                m[c < min_conf] = 0.0
        depths.append(m.astype(np.float16))

    idx = np.minimum((np.arange(n_video) * len(depths) / n_video).astype(int), len(depths) - 1)
    stack = np.stack([depths[i] for i in idx])
    out = folder / "rgb.mp4.depth.npz"
    np.savez_compressed(out, depth=stack, ts=np.arange(n_video) / (fps or 30.0))
    valid = stack[stack > 0]
    print(f"{video}: {n_video} video frames @ {fps:.1f} fps, {len(depths)} depth frames {depths[0].shape}; "
          f"depth {valid.min():.2f}-{valid.max():.2f} m, {valid.size / stack.size:.0%} valid -> {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    convert(Path(sys.argv[1]))
