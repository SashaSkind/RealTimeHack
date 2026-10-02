"""Test B follow-up: which single prompt words and input sizes work per category?
Top-1 loose hit (IoU>=0.3 or centre inside label) on positives, and neg_fire on unrelated
categories, at conf 0.05 and 0.1."""
import csv, json, sys, time
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.detector_bench import load_labels, matches, RELATED, PHOTOS
from handguide.target import OpenVocabDetector

CANDS = {
    "lever_handle": ["door handle", "handle", "lever handle", "door lever", "lever", "doorknob", "door handle lever"],
    "door_knob": ["door knob", "doorknob", "knob", "handle", "door handle", "round knob"],
    "cabinet_handle": ["cabinet handle", "handle", "drawer handle", "drawer pull", "knob", "pull handle"],
    "light_switch": ["light switch", "switch", "wall switch", "switch plate", "electrical switch", "wall plate"],
    "cup": ["cup", "mug", "coffee cup"],
}
rows = load_labels()
imgs = {r["file"]: cv2.cvtColor(cv2.imread(str(PHOTOS / r["file"])), cv2.COLOR_BGR2RGB) for r in rows}
res = {}
for kind in sys.argv[1].split(","):
    for imgsz in (640, 1024):
        det = OpenVocabDetector(kind, device="mps", conf=0.01, imgsz=imgsz)
        for cat, cands in CANDS.items():
            pos = [r for r in rows if r["category"] == cat]
            neg = [r for r in rows if r["category"] != cat and r["category"] not in RELATED.get(cat, set())]
            for p in cands:
                det.set_prompts([p])
                top = {r["file"]: (det.detect(imgs[r["file"]]) or [None])[0] for r in pos + neg}
                for thr in (0.05, 0.1):
                    hit = sum(1 for r in pos if top[r["file"]] and top[r["file"]].score >= thr and matches(top[r["file"]].box, r["boxes"], loose=True))
                    fire = sum(1 for r in neg if top[r["file"]] and top[r["file"]].score >= thr)
                    res[f"{kind}|{imgsz}|{cat}|{p}|{thr}"] = (hit / len(pos), fire / len(neg))
            print(kind, imgsz, cat, {p: res[f"{kind}|{imgsz}|{cat}|{p}|0.05"] for p in cands}, flush=True)
json.dump(res, open(ROOT / "output/bench/prompt_search.json", "w"), indent=0)
