"""Test B: YOLO-World vs YOLOE on labelled Commons photos.

For each model x prompt-set x category we set ONLY that category's prompts (that's how
the app runs: it knows its target), run every photo, and store all detections >= 0.01.
Metrics are then computed at several confidence thresholds:

- hit@0.5:     top-scoring detection on a positive image has IoU >= 0.5 with a labelled box
- hit_loose:   top detection IoU >= 0.3 OR its centre lies inside a labelled box
               (labels are rough; switch boxes cover the whole plate)
- fp_pos:      mean number of extra detections on positive images that match no labelled box
               (images with many unlabelled instances are excluded)
- neg_fire:    fraction of images from unrelated categories where the prompt fires at all
- speed:       median ms per image at imgsz 640 on mps and cpu

Usage: python bench/detector_bench.py [--models yoloe,yoloworld,...] [--devices mps,cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handguide.target import OpenVocabDetector, iou  # noqa: E402

PHOTOS = ROOT / "data" / "photos"
OUT = ROOT / "output" / "bench"

PROMPTS = {
    "single": {
        "lever_handle": ["door handle"],
        "door_knob": ["door knob"],
        "cabinet_handle": ["cabinet handle"],
        "light_switch": ["light switch"],
        "cup": ["cup"],
    },
    "multi": {
        "lever_handle": ["door handle", "lever door handle", "door lever"],
        "door_knob": ["door knob", "doorknob", "round door knob"],
        "cabinet_handle": ["cabinet handle", "drawer handle", "drawer pull"],
        "light_switch": ["light switch", "wall switch", "switch plate"],
        "cup": ["cup", "mug", "coffee cup"],
    },
    # chosen from bench/prompt_search.py (single-prompt sweep), see report
    "tuned": {
        "lever_handle": ["handle", "lever handle"],
        "door_knob": ["round knob", "door knob"],
        "cabinet_handle": ["cabinet handle", "drawer handle"],
        "light_switch": ["switch", "switch plate"],
        "cup": ["cup", "mug"],
    },
}
# Categories that should not count as negatives for each other.
RELATED = {"lever_handle": {"door_knob"}, "door_knob": {"lever_handle"}}
NON_EXHAUSTIVE = {"cabinet_handle_08.jpg", "cabinet_handle_09.jpg"}
THRESHOLDS = [0.05, 0.1, 0.2, 0.3]


def load_labels():
    rows = list(csv.DictReader(open(PHOTOS / "labels.csv")))
    for r in rows:
        r["boxes"] = [tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))] + \
                     [tuple(map(float, b)) for b in json.loads(r["extra_boxes"] or "[]")]
    return rows


def matches(det_box, gt_boxes, loose=False):
    best = max(iou(det_box, g) for g in gt_boxes)
    if not loose:
        return best >= 0.5
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    inside = any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gt_boxes)
    return best >= 0.3 or inside


def run_model(kind, rows, images, device):
    det = OpenVocabDetector(kind, device=device, conf=0.01, imgsz=640)
    raw = {}
    for variant, pmap in PROMPTS.items():
        for cat, prompts in pmap.items():
            det.set_prompts(prompts)
            det.detect(images[rows[0]["file"]])  # warm-up after prompt change
            for r in rows:
                t = time.perf_counter()
                ds = det.detect(images[r["file"]])
                ms = (time.perf_counter() - t) * 1000
                raw[f"{variant}|{cat}|{r['file']}"] = {"ms": ms, "dets": [[*d.box, d.score, d.label] for d in ds[:20]]}
    return raw


def speed(kind, rows, images, device, n=30):
    det = OpenVocabDetector(kind, device=device, conf=0.1, imgsz=640)
    det.set_prompts(PROMPTS["single"]["cup"])
    for r in rows[:3]:
        det.detect(images[r["file"]])
    ts = []
    for r in (rows * 2)[:n]:
        t = time.perf_counter()
        det.detect(images[r["file"]])
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def metrics(raw, rows, variant, thr):
    out = {}
    for cat in PROMPTS[variant]:
        pos = [r for r in rows if r["category"] == cat]
        neg = [r for r in rows if r["category"] != cat and r["category"] not in RELATED.get(cat, set())]
        hit = loose = fp = fp_n = 0
        misses = []
        for r in pos:
            ds = [d for d in raw[f"{variant}|{cat}|{r['file']}"]["dets"] if d[4] >= thr]
            if ds and matches(ds[0][:4], r["boxes"]):
                hit += 1
            if ds and matches(ds[0][:4], r["boxes"], loose=True):
                loose += 1
            else:
                misses.append(r["file"])
            if r["file"] not in NON_EXHAUSTIVE:
                fp += sum(1 for d in ds if not matches(d[:4], r["boxes"], loose=True))
                fp_n += 1
        fire = sum(1 for r in neg if any(d[4] >= thr for d in raw[f"{variant}|{cat}|{r['file']}"]["dets"]))
        out[cat] = {"n": len(pos), "hit": hit / len(pos), "hit_loose": loose / len(pos),
                    "fp_per_img": fp / max(1, fp_n), "neg_fire": fire / max(1, len(neg)), "n_neg": len(neg),
                    "misses": misses}
    tot = {k: float(np.mean([v[k] for v in out.values()])) for k in ("hit", "hit_loose", "fp_per_img", "neg_fire")}
    return out, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yoloworld,yoloworld-m,yoloe,yoloe-m,yoloe26")
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--device", default="mps", help="device for the accuracy runs")
    ap.add_argument("--speed-json", default=None, help="take speeds from this JSON {model: {dev: [median, p95]}}")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_labels()
    images = {r["file"]: cv2.cvtColor(cv2.imread(str(PHOTOS / r["file"])), cv2.COLOR_BGR2RGB) for r in rows}

    results = {}
    for kind in args.models.split(","):
        print(f"== {kind}", flush=True)
        t0 = time.time()
        raw = run_model(kind, rows, images, args.device)
        json.dump(raw, open(OUT / f"detector_raw_{kind}.json", "w"))
        sp = {}
        for dev in [d for d in args.devices.split(",") if d]:
            try:
                sp[dev] = speed(kind, rows, images, dev)
            except Exception as e:
                sp[dev] = (float("nan"), float("nan"))
                print(f"   speed on {dev} failed: {e}")
        if args.speed_json:  # reuse speeds measured earlier under clean conditions
            sp = {k: tuple(v) for k, v in json.load(open(args.speed_json)).get(kind, {}).items()}
        results[kind] = {"speed": sp, "metrics": {}}
        for variant in PROMPTS:
            for thr in THRESHOLDS:
                per, tot = metrics(raw, rows, variant, thr)
                results[kind]["metrics"][f"{variant}@{thr}"] = {"per_cat": per, "total": tot}
        print(f"   done in {time.time() - t0:.0f}s speed={sp}", flush=True)
    json.dump(results, open(OUT / "detector_results.json", "w"), indent=1)
    write_md(results)


def write_md(results):
    L = ["# Test B: detector benchmark", "",
         "48 Commons photos, 5 categories (see data/photos/labels.csv). Only the target category's prompts are active per run.",
         "hit = top detection IoU>=0.5; loose = top detection IoU>=0.3 or centre inside label; "
         "fp/img = unmatched detections on positive images; neg_fire = share of unrelated-category images where the prompt fires.", ""]
    L += ["## Summary (mean over categories)", "",
          "| model | prompts@thr | hit | loose | fp/img | neg_fire | mps ms (med/p95) | cpu ms (med/p95) |", "|---|---|---|---|---|---|---|---|"]
    for kind, R in results.items():
        sp = R["speed"]
        for key, M in R["metrics"].items():
            t = M["total"]
            L.append(f"| {kind} | {key} | {t['hit']:.2f} | {t['hit_loose']:.2f} | {t['fp_per_img']:.2f} | {t['neg_fire']:.2f} | "
                     f"{sp.get('mps', (np.nan,))[0]:.0f}/{sp.get('mps', (np.nan, np.nan))[1]:.0f} | "
                     f"{sp.get('cpu', (np.nan,))[0]:.0f}/{sp.get('cpu', (np.nan, np.nan))[1]:.0f} |")
    L += ["", "## Per category", ""]
    for kind, R in results.items():
        for key, M in R["metrics"].items():
            L += [f"### {kind} {key}", "", "| category | n | hit | loose | fp/img | neg_fire | loose misses |", "|---|---|---|---|---|---|---|"]
            for cat, v in M["per_cat"].items():
                L.append(f"| {cat} | {v['n']} | {v['hit']:.2f} | {v['hit_loose']:.2f} | {v['fp_per_img']:.2f} | {v['neg_fire']:.2f} | {', '.join(m[:-4] for m in v['misses'])} |")
            L.append("")
    (OUT / "detector_results.md").write_text("\n".join(L))
    print("wrote", OUT / "detector_results.md")


if __name__ == "__main__":
    main()
