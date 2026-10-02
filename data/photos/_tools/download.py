"""Download selected candidates as <category>_<NN>.jpg (max width 1024) and record metadata.

Usage: python download.py <category> <cand_idx> [<cand_idx> ...]
Appends/updates _tools/meta.json keyed by file name. Numbering continues after existing files.
Commons now serves thumbnails only at standard widths (e.g. 1280), so the API's thumburl for
iiurlwidth=1024 is a 1280px (or unscaled original) file; we downscale locally to 1024 wide.
"""
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

UA = "HandGuideBench/0.1 (research; https://github.com/SashaSkind/RealTimeHack)"
ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "_tools" / "meta.json"
S = requests.Session()
S.headers["User-Agent"] = UA


def main():
    cat = sys.argv[1]
    idxs = [int(x) for x in sys.argv[2:]]
    cands = json.loads((ROOT / "_cand" / f"candidates_{cat}.json").read_text())
    meta = json.loads(META.read_text()) if META.exists() else {}
    used_titles = {m["title"] for m in meta.values()}
    existing = [int(k.rsplit("_", 1)[1][:2]) for k in meta if k.startswith(cat + "_")]
    n = max(existing, default=0)
    for i in idxs:
        d = cands[i]
        if d["title"] in used_titles:
            print("skip dup", i, d["title"])
            continue
        url = d["thumb"]
        for attempt in range(4):
            r = S.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if im.width > 1024:
            im = im.resize((1024, round(im.height * 1024 / im.width)), Image.LANCZOS)
        n += 1
        fn = f"{cat}_{n:02d}.jpg"
        im.save(ROOT / fn, quality=92)
        meta[fn] = dict(cand_idx=i, category=cat, title=d["title"], page=d["page"], image_url=url,
                        license=d["license"], author=d["author"], width=im.width, height=im.height)
        print(fn, im.size, i, d["title"])
        META.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
