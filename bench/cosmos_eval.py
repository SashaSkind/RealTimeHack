"""Test C: Cosmos Reason 2 slow-loop prompt on the door-handle, door-knob and cup photos.

Uses the exact client the app uses (handguide.slow_loop.CosmosClient) against any
OpenAI-compatible server. Records per image: JSON parsed?, target, affordance,
instruction, box (IoU vs the rough label under the requested box format, plus IoU if the
same numbers are read the other way), seconds per call, and server-side timings if the
server reports them (llama-server does).

Usage:
  python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag 8b_prompt --json-mode prompt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handguide.slow_loop import CosmosClient, box_to_pixels, encode_jpeg  # noqa: E402
from handguide.target import iou  # noqa: E402

PHOTOS = ROOT / "data" / "photos"
OUT = ROOT / "output" / "bench" / "cosmos"

GOALS = {"lever_handle": "open the door", "door_knob": "open the door", "cup": "pick up the cup",
         "cabinet_handle": "open the cabinet drawer", "light_switch": "turn on the light"}
TARGET_WORDS = {"lever_handle": ["handle", "lever"], "door_knob": ["knob", "handle"],
                "cup": ["cup", "mug"], "cabinet_handle": ["handle", "pull", "knob"], "light_switch": ["switch"]}
AFF_WORDS = {"lever_handle": ["press", "push down", "pull down", "down"],
             "door_knob": ["turn", "twist", "rotate"],
             "cup": ["grab", "grasp", "pick", "hold", "lift", "handle"],
             "cabinet_handle": ["pull"], "light_switch": ["flip", "press", "push", "toggle", "switch"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="cosmos")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json-mode", default="prompt", choices=["prompt", "schema", "object"])
    ap.add_argument("--box-format", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--categories", default="lever_handle,door_knob,cup")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--no-hints", action="store_true", help="leave out the affordance hint sentence")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cats = args.categories.split(",")
    rows = [r for r in csv.DictReader(open(PHOTOS / "labels.csv")) if r["category"] in cats]
    if args.limit:
        rows = rows[:args.limit]
    client = CosmosClient(args.url, model=args.model, json_mode=args.json_mode, box_format=args.box_format,
                          timeout_s=120, max_side=args.max_side, hints=not args.no_hints)
    other = "pixel" if args.box_format == "norm1000" else "norm1000"

    # warm-up call (not recorded)
    client.ask(cv2.cvtColor(cv2.imread(str(PHOTOS / rows[0]["file"])), cv2.COLOR_BGR2RGB), "warm up")

    results = []
    for r in rows:
        rgb = cv2.cvtColor(cv2.imread(str(PHOTOS / r["file"])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gts = [tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))] + \
              [tuple(map(float, b)) for b in json.loads(r["extra_boxes"] or "[]")]
        res = client.ask(rgb, GOALS[r["category"]])
        _, sent_wh = encode_jpeg(rgb, args.max_side)
        iou_main = max((iou(res.box, g) for g in gts), default=0.0) if res.box else None
        alt = box_to_pixels(res.box_raw, other, sent_wh, (w, h)) if res.box_raw else None
        iou_alt = max((iou(alt, g) for g in gts), default=0.0) if alt else None
        tl, al = res.target.lower(), res.affordance.lower()
        row = {
            "file": r["file"], "category": r["category"], "parsed": res.ok, "error": res.error,
            "seconds": round(res.latency_s, 2), "target": res.target, "affordance": res.affordance,
            "instruction": res.instruction, "phase": res.phase, "box_raw": res.box_raw,
            "iou": None if iou_main is None else round(iou_main, 3),
            f"iou_if_{other}": None if iou_alt is None else round(iou_alt, 3),
            "target_kw": any(k in tl for k in TARGET_WORDS[r["category"]]),
            "affordance_kw": any(k in al for k in AFF_WORDS[r["category"]]),
            "raw": res.raw,
        }
        results.append(row)
        print(f"{r['file']:22s} ok={res.ok} {res.latency_s:5.2f}s iou={row['iou']} alt={row[f'iou_if_{other}']} "
              f"| {res.target} | {res.affordance} | {res.instruction}", flush=True)

    json.dump(results, open(OUT / f"{args.tag}.json", "w"), indent=1)
    ok = [x for x in results if x["parsed"]]
    secs = [x["seconds"] for x in results]
    ious = [x["iou"] or 0.0 for x in ok]
    summ = {
        "tag": args.tag, "model": args.model, "json_mode": args.json_mode, "box_format": args.box_format,
        "n": len(results), "parsed": len(ok),
        "target_kw": sum(x["target_kw"] for x in ok), "affordance_kw": sum(x["affordance_kw"] for x in ok),
        "box_returned": sum(1 for x in ok if x["box_raw"]),
        "iou_ge_0.5": sum(1 for v in ious if v >= 0.5), "iou_ge_0.3": sum(1 for v in ious if v >= 0.3),
        "median_iou": float(np.median(ious)) if ious else None,
        f"iou_ge_0.5_if_{other}": sum(1 for x in ok if (x[f"iou_if_{other}"] or 0) >= 0.5),
        "sec_median": float(np.median(secs)), "sec_p95": float(np.percentile(secs, 95)), "sec_max": float(max(secs)),
    }
    json.dump(summ, open(OUT / f"{args.tag}_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
