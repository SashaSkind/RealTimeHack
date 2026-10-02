"""Drop rejected images, renumber per category, regenerate _grid/_check, write labels.csv.

Usage: python finalize.py <dropped_file.jpg> [...]
Dropped images (and any downloaded image without a box) are moved to _dropped/.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "_tools"
CATS = ["lever_handle", "door_knob", "cabinet_handle", "light_switch", "cup"]


def main():
    drop = set(sys.argv[1:])
    meta = json.loads((T / "meta.json").read_text())
    boxes = json.loads((T / "boxes.json").read_text())
    (ROOT / "_dropped").mkdir(exist_ok=True)
    for fn in list(meta):
        if fn in drop or fn not in boxes:
            if (ROOT / fn).exists():
                shutil.move(ROOT / fn, ROOT / "_dropped" / fn)
            meta.pop(fn)
            boxes.pop(fn, None)
    for d in ("_grid", "_check"):
        for p in (ROOT / d).glob("*.jpg"):
            if not p.name.startswith("sheet_"):
                p.unlink()
    new_meta, new_boxes, rows = {}, {}, []
    tmp = ROOT / "_renaming"
    tmp.mkdir(exist_ok=True)
    for cat in CATS:
        files = sorted(f for f in meta if meta[f]["category"] == cat)
        for k, old in enumerate(files, 1):
            new = f"{cat}_{k:02d}.jpg"
            shutil.move(ROOT / old, tmp / new)
            new_meta[new] = meta[old]
            new_boxes[new] = boxes[old]
    for p in tmp.iterdir():
        shutil.move(p, ROOT / p.name)
    tmp.rmdir()
    (T / "meta.json").write_text(json.dumps(new_meta, indent=1, ensure_ascii=False))
    (T / "boxes.json").write_text(json.dumps(new_boxes, indent=1))
    names = list(new_meta)
    subprocess.run([sys.executable, str(T / "grid.py"), *names], check=True, stdout=subprocess.DEVNULL)
    for n in names:
        subprocess.run([sys.executable, str(T / "check.py"), n], check=True, stdout=subprocess.DEVNULL)
    for n in names:
        m, b = new_meta[n], new_boxes[n]
        x1, y1, x2, y2 = b["box"]
        frac = (x2 - x1) * (y2 - y1) / (m["width"] * m["height"])
        rows.append(dict(file=n, category=m["category"], x1=x1, y1=y1, x2=x2, y2=y2,
                         extra_boxes=json.dumps(b["extra"]) if b["extra"] else "",
                         scene="closeup" if frac > 0.15 else "cluttered",
                         width=m["width"], height=m["height"], source_page_url=m["page"],
                         image_url=m["image_url"], license=m["license"], author=m["author"]))
    with open(ROOT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
