"""Test E: open-loop sanity check on a recorded reach.

For each frame in the approach window, compare the cue direction (dx, dy) the system would
play with the direction the hand actually moved over the next k frames. The person in the
clip heard nothing; they just reached for the object. If our cue points the way a correct
reach goes, the agreement rate is high. A sign flip (e.g. image y not inverted) drives it
toward 0, so this catches sign and axis bugs.

It does NOT show a blind person can follow the cues. That needs the live test.

Usage:
  python bench/open_loop_check.py --clip data/clips/reach_bottle.ogv --target bottle
  python bench/open_loop_check.py --clip data/clips/grasp_lift_trial.ogv --box 780,530,850,645@200 --approach 200-236
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handguide.guidance import GuidanceConfig, Guide, compute_cue  # noqa: E402
from handguide.hand import HandTracker  # noqa: E402
from handguide.target import OpenVocabDetector, TargetTracker  # noqa: E402


def rotate_box(box, rot, w_new, h_new):
    """Map a box from the original frame into the frame rotated clockwise by `rot`."""
    x1, y1, x2, y2 = box
    if rot == 90:    # original H = w_new
        return (w_new - y2, x1, w_new - y1, x2)
    if rot == 270:   # original W = h_new
        return (y1, h_new - x2, y2, h_new - x1)
    if rot == 180:
        return (w_new - x2, h_new - y2, w_new - x1, h_new - y1)
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--target", default=None, help="detector prompt (e.g. cup, bottle)")
    ap.add_argument("--box", default=None, help="x1,y1,x2,y2@frame: initial target box, tracked with CSRT")
    ap.add_argument("--approach", default=None, help="frame range a-b of the reach (default: whole clip)")
    ap.add_argument("--k", type=int, default=3, help="look-ahead frames for hand motion")
    ap.add_argument("--min-move", type=float, default=0.004, help="min motion per frame, fraction of frame width")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate the clip clockwise; turns a horizontal reach into a vertical one to exercise the y axis")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = Path(args.clip).stem + (f"_rot{args.rotate}" if args.rotate else "")
    out = Path(args.out or ROOT / "output" / "bench" / "open_loop" / name)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if args.rotate:
            f = cv2.rotate(f, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[args.rotate])
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    H, W = frames[0].shape[:2]

    a, b = (0, len(frames) - 1)
    if args.approach:
        a, b = map(int, args.approach.split("-"))
        b = min(b, len(frames) - 1)

    hand = HandTracker(mode="video")
    if args.box:
        coords, f0 = args.box.split("@")
        box0, f0 = tuple(float(v) for v in coords.split(",")), int(f0)
        box0 = rotate_box(box0, args.rotate, frames[0].shape[1], frames[0].shape[0])
        tt = TargetTracker(None, target="x", detect_every=10 ** 9, track_scale=1.0)
    else:
        det = OpenVocabDetector("yoloe", conf=0.1)
        tt = TargetTracker(det, target=args.target, detect_every=5, track_scale=0.5)
        f0 = 0

    palms, boxes = {}, {}
    prev_box = None
    for i, rgb in enumerate(frames):
        h = hand.detect(rgb, int(i * 1000 / fps),
                        prefer_near=None if prev_box is None else ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2))
        if h.found:
            palms[i] = (h.x, h.y)
        if args.box and i == f0:
            tt.set_external_box(box0)
        if i >= f0:
            s = tt.update(rgb)
            if s.box is not None:
                boxes[i] = s.box
                prev_box = s.box

    guide = Guide(GuidanceConfig())
    rows = []
    for i in range(a, b + 1):
        p, q = palms.get(i), palms.get(i + args.k)
        box = boxes.get(i)
        smooth = guide.update(p, box, (W, H))
        if p is None or q is None or box is None:
            continue
        vx, vy = (q[0] - p[0]), (q[1] - p[1])
        speed = np.hypot(vx, vy) / args.k / W
        cue = compute_cue(p, box, (W, H))
        mx, my = vx, -vy  # motion in cue coordinates (up positive)
        rows.append(dict(frame=i, hand_x=p[0], hand_y=p[1], move_x=vx, move_y=vy, speed=speed,
                         dx=cue.dx, dy=cue.dy, dist=cue.distance, state=cue.state,
                         sdx=smooth.dx, sdy=smooth.dy))

    moving = [r for r in rows if r["speed"] >= args.min_move and (r["dx"] or r["dy"])]

    def agree(rs, flip_y=False, key=("dx", "dy")):
        ok = []
        for r in rs:
            cx, cy = r[key[0]], (-r[key[1]] if flip_y else r[key[1]])
            mx, my = r["move_x"], -r["move_y"]
            ok.append(cx * mx + cy * my > 0)
        return float(np.mean(ok)) if ok else float("nan")

    def axis(rs, ax):
        ok = []
        for r in rs:
            c = r["dx"] if ax == "x" else r["dy"]
            m = r["move_x"] if ax == "x" else -r["move_y"]
            if c != 0 and abs(m) / args.k / W >= args.min_move / 2:
                ok.append(np.sign(c) == np.sign(m))
        return (float(np.mean(ok)) if ok else float("nan")), len(ok)

    ax_x, n_x = axis(moving, "x")
    ax_y, n_y = axis(moving, "y")
    summ = {
        "clip": args.clip, "frames": len(frames), "approach": [a, b], "fps": fps,
        "hand_found_in_window": sum(1 for i in range(a, b + 1) if i in palms),
        "target_found_in_window": sum(1 for i in range(a, b + 1) if i in boxes),
        "window_frames": b - a + 1, "scored_frames": len(moving),
        "agreement": agree(moving), "agreement_smoothed_cue": agree(moving, key=("sdx", "sdy")),
        "x_sign_agreement": ax_x, "x_n": n_x, "y_sign_agreement": ax_y, "y_n": n_y,
        "control_agreement_if_y_flipped": agree(moving, flip_y=True),
    }
    with open(out / "frames.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    # trajectory picture: palm path, target box, cue arrows
    img = cv2.cvtColor(frames[min(b, len(frames) - 1)], cv2.COLOR_RGB2BGR)
    for i in range(a, b + 1):
        if i in palms and i + 1 in palms:
            cv2.line(img, tuple(map(int, palms[i])), tuple(map(int, palms[i + 1])), (255, 0, 255), 2)
    for r in moving[::2]:
        p = (int(r["hand_x"]), int(r["hand_y"]))
        cv2.arrowedLine(img, p, (int(p[0] + 40 * r["dx"]), int(p[1] - 40 * r["dy"])), (0, 255, 0), 2, tipLength=0.3)
    if a in boxes:
        x1, y1, x2, y2 = map(int, boxes[a])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imwrite(str(out / "trajectory.jpg"), img)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
