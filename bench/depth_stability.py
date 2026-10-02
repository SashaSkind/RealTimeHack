"""Live test 2 helper: how stable is palm depth while the hand is held still?
Reads a run's cues.csv and reports, over sliding 2 s windows where the hand is found,
the std and peak-to-peak of the (smoothed) palm depth. Pass: the quietest multi-second
window has peak-to-peak under ~3 cm with LiDAR (depth_kind == metric).

  python bench/depth_stability.py output/runs/live_r3d/cues.csv
"""
import csv, sys
import numpy as np

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["hand_depth"] not in ("", "None")]
if not rows:
    sys.exit("no frames with palm depth; was the hand visible and depth on?")
kinds = {r["depth_kind"] for r in rows}
t = np.array([float(r["t"]) for r in rows])
d = np.array([float(r["hand_depth"]) for r in rows])
print(f"{len(rows)} frames with palm depth, kind={kinds}, range {d.min():.3f}..{d.max():.3f}")
win = 2.0
best = None
for i in range(len(t)):
    m = (t >= t[i]) & (t < t[i] + win)
    if m.sum() < 20:
        continue
    seg = d[m]
    s = (seg.std(), np.ptp(seg), t[i])
    best = s if best is None or s[1] < best[1] else best
if best:
    unit = "m" if kinds == {"metric"} else "(relative units)"
    print(f"quietest {win:.0f}s window at t={best[2]:.1f}s: std={best[0]:.4f} {unit}, peak-to-peak={best[1]:.4f} {unit}")
    if kinds == {"metric"}:
        print("PASS" if best[1] < 0.03 else "FAIL", "(threshold: peak-to-peak < 3 cm)")
