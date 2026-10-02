"""First contact with the iPhone over Record3D: grab ~3 s of frames and report what arrives.

  python bench/r3d_probe.py             # Record3D app (USB)
  python bench/r3d_probe.py pathon      # free "Record3DStream - PathOn" app over USB
  python bench/r3d_probe.py pathon IP   # PathOn over Wi-Fi
Prints stats and saves output/r3d_probe.jpg (RGB | depth).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handguide.sources import PathOnSource, Record3DSource, SourceError  # noqa: E402

try:
    if len(sys.argv) > 1 and sys.argv[1] == "pathon":
        src = PathOnSource(host=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        src = Record3DSource(0)
except SourceError as e:
    sys.exit(f"[probe] {e}")

t0, n, last = time.perf_counter(), 0, None
for fr in src:
    n += 1
    last = fr
    if time.perf_counter() - t0 > 3.0:
        break
dt = time.perf_counter() - t0
src.close()
if last is None:
    sys.exit("[probe] connected but no frames arrived; press the red button in Record3D")

rgb, d = last.rgb, last.depth
valid = d[(d > 0) & np.isfinite(d)]
h, w = d.shape
c = d[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]
c = c[c > 0]
print(f"frames: {n} in {dt:.1f}s = {n / dt:.1f} fps")
print(f"rgb: {rgb.shape}  depth: {d.shape}  device: {src.stats.extra}")
print(f"depth valid: {valid.size / d.size:.0%}  range {valid.min():.2f}-{valid.max():.2f} m  "
      f"centre {np.median(c) if c.size else float('nan'):.3f} m")
if hasattr(src, "intrinsics"):
    print(f"intrinsics:\n{src.intrinsics()}")

vis = cv2.applyColorMap((np.clip(d / 3.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
vis = cv2.resize(vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
out = np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), vis])
cv2.imwrite(str(ROOT / "output" / "r3d_probe.jpg"), cv2.resize(out, None, fx=0.5, fy=0.5))
print("saved output/r3d_probe.jpg (left: RGB, right: depth 0-3 m)")
