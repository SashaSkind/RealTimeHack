"""Build a contact sheet of candidate thumbnails for quick visual screening.

Usage: python sheet.py <category> <start> <count> [title_regex]  (start/count index into filtered list)
Writes _cand/sheet_<category>_<start>.jpg; tile label = candidate index.
"""
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

UA = "HandGuideBench/0.1 (research; https://github.com/SashaSkind/RealTimeHack)"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "_cand" / "thumbs"
CACHE.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = UA
TILE = 300


def small_url(u):
    import re
    return re.sub(r"/(\d+)px-", "/330px-", u)


def get_thumb(cat, i, d):
    p = CACHE / f"{cat}_{i}.jpg"
    if not p.exists():
        u = small_url(d["thumb"])
        for attempt in range(3):
            r = S.get(u, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((TILE, TILE))
        im.save(p, quality=85)
        time.sleep(0.5)
    return cv2.imread(str(p))


def main():
    cat, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cands = json.loads((ROOT / "_cand" / f"candidates_{cat}.json").read_text())
    cols = 6
    import re
    rx = re.compile(sys.argv[4], re.I) if len(sys.argv) > 4 else None
    allidx = [i for i, d in enumerate(cands) if rx is None or rx.search(d["title"])]
    print("filtered", len(allidx))
    idx = allidx[start:start + count]
    rows = (len(idx) + cols - 1) // cols
    sheet = np.full((rows * TILE, cols * TILE, 3), 255, np.uint8)
    for k, i in enumerate(idx):
        try:
            im = get_thumb(cat, i, cands[i])
        except Exception as e:
            print("fail", i, e)
            continue
        r, c = divmod(k, cols)
        h, w = im.shape[:2]
        sheet[r * TILE:r * TILE + h, c * TILE:c * TILE + w] = im
        cv2.rectangle(sheet, (c * TILE, r * TILE), (c * TILE + 60, r * TILE + 34), (0, 0, 0), -1)
        cv2.putText(sheet, str(i), (c * TILE + 4, r * TILE + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    out = ROOT / "_cand" / f"sheet_{cat}_{start}.jpg"
    cv2.imwrite(str(out), sheet)
    print(out)


if __name__ == "__main__":
    main()
