"""Record a box for an image and render a verification image to _check/.

Usage:
  python check.py <file.jpg> x1 y1 x2 y2 [extra_json]   # set box (+ optional extra boxes) and render
  python check.py <file.jpg>                            # re-render from stored box
Boxes stored in _tools/boxes.json. Main box drawn red, extras blue.
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
BOXES = ROOT / "_tools" / "boxes.json"


def main():
    fn = sys.argv[1]
    boxes = json.loads(BOXES.read_text()) if BOXES.exists() else {}
    if len(sys.argv) >= 6:
        box = [int(v) for v in sys.argv[2:6]]
        extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else []
        boxes[fn] = dict(box=box, extra=extra)
        BOXES.write_text(json.dumps(boxes, indent=1))
    b = boxes[fn]
    im = cv2.imread(str(ROOT / fn))
    h, w = im.shape[:2]
    x1, y1, x2, y2 = b["box"]
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, (b["box"], w, h)
    t = max(2, w // 300)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), t)
    for e in b["extra"]:
        cv2.rectangle(im, (e[0], e[1]), (e[2], e[3]), (255, 128, 0), t)
    frac = (x2 - x1) * (y2 - y1) / (w * h)
    cv2.putText(im, f"{frac:.1%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(str(ROOT / "_check" / fn), im)
    print(fn, b, f"area={frac:.1%}")


if __name__ == "__main__":
    main()
