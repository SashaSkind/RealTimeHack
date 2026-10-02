"""Quick fixed micro-benchmark to detect throttling: run before/after timing-sensitive tests and
compare with the clean reference (20:30-21:05 on 2026-09-29, M3 Pro): YOLOE-11s MPS ~27 ms,
MediaPipe hand ~30 ms, Depth Anything V2 Small (252 px, MPS) ~28 ms."""
import sys, time, json
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from handguide.target import OpenVocabDetector
from handguide.hand import HandTracker
from handguide.depth import DepthAnything

cap = cv2.VideoCapture(str(ROOT / "data/clips/reach_bottle.ogv"))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(cv2.resize(f, (960, 600)), cv2.COLOR_BGR2RGB))

def med(fn, n=30):
    for i in range(3):
        fn(i)
    ts = []
    for i in range(n):
        t = time.perf_counter(); fn(i); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 1)

det = OpenVocabDetector("yoloe", conf=0.1); det.set_prompts(["bottle"])
hand = HandTracker(mode="image")
dav2 = DepthAnything()
res = {"t": time.strftime("%H:%M:%S"),
       "yoloe_ms": med(lambda i: det.detect(frames[i % len(frames)])),
       "hand_ms": med(lambda i: hand.detect(frames[i % len(frames)])),
       "dav2_ms": med(lambda i: dav2(frames[i % len(frames)]))}
import subprocess
res["battery"] = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.split("\t")[-1].split(";")[0]
print(json.dumps(res))
with open(ROOT / "output/bench/probe_log.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
