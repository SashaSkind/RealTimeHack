"""Query Wikimedia Commons for candidate images per category, filter by license/mime.

Writes _cand/candidates_<category>.json (list of dicts).
Usage: python search.py <category> [max]
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "HandGuideBench/0.1 (research; https://github.com/SashaSkind/RealTimeHack)"
S = requests.Session()
S.headers["User-Agent"] = UA
ROOT = Path(__file__).resolve().parent.parent

SOURCES = {
    "lever_handle": [
        ("cat", "Category:Lever handles"),
        ("cat", "Category:Door handles"),
        ("search", "door lever handle"),
        ("search", "door handle lever interior"),
    ],
    "door_knob": [
        ("cat", "Category:Doorknobs"),
        ("cat", "Category:Door knobs"),
        ("search", "doorknob door"),
        ("search", "round door knob"),
    ],
    "cabinet_handle": [
        ("cat", "Category:Drawer pulls"),
        ("cat", "Category:Cabinet handles"),
        ("cat", "Category:Drawer handles"),
        ("search", "drawer handle kitchen cabinet"),
        ("search", "cabinet pull handle"),
        ("cat", "Category:Kitchen cabinets"),
        ("search", "kitchen cabinet handles"),
        ("search", "kitchen drawers handle"),
    ],
    "light_switch": [
        ("cat", "Category:Light switches"),
        ("search", "light switch wall"),
        ("search", "rocker light switch"),
        ("search", "toggle light switch"),
    ],
    "cup": [
        ("cat", "Category:Mugs"),
        ("cat", "Category:Coffee cups"),
        ("cat", "Category:Cups"),
        ("search", "coffee mug table"),
        ("search", "cup of coffee"),
        ("search", "mug on desk"),
    ],
}

OK_LICENSE = re.compile(r"^(CC0|Public domain|PD|CC BY(-SA)? [0-9.]+|CC BY(-SA)?$|CC-BY|CC BY-SA)", re.I)
BAD = re.compile(r"\b(NC|ND)\b|fair use|non-commercial", re.I)


def license_ok(lic):
    if not lic:
        return False
    if BAD.search(lic):
        return False
    return bool(OK_LICENSE.match(lic.strip()))


def strip_html(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def query(params):
    base = dict(action="query", format="json", prop="imageinfo",
                iiprop="url|extmetadata|size|mime", iiurlwidth=1024)
    base.update(params)
    out = []
    cont = {}
    for _ in range(4):
        r = S.get(API, params={**base, **cont}, timeout=30)
        r.raise_for_status()
        j = r.json()
        out.extend((j.get("query") or {}).get("pages", {}).values())
        if "continue" not in j:
            break
        cont = j["continue"]
        time.sleep(0.3)
    return out


def main():
    cat = sys.argv[1]
    seen = {}
    for kind, q in SOURCES[cat]:
        if kind == "cat":
            pages = query(dict(generator="categorymembers", gcmtitle=q, gcmtype="file", gcmlimit=100))
        else:
            pages = query(dict(generator="search", gsrnamespace=6, gsrsearch=q, gsrlimit=50))
        n = 0
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            em = ii.get("extmetadata", {})
            lic = (em.get("LicenseShortName") or {}).get("value", "")
            if not license_ok(lic):
                continue
            if not ii.get("thumburl"):
                continue
            t = p["title"]
            if t in seen:
                continue
            seen[t] = dict(
                title=t, source=q, mime=mime, license=lic,
                author=strip_html((em.get("Artist") or {}).get("value", "")),
                desc=strip_html((em.get("ImageDescription") or {}).get("value", ""))[:200],
                page=ii.get("descriptionurl"), thumb=ii["thumburl"],
                w=ii.get("width"), h=ii.get("height"),
            )
            n += 1
        print(f"{q}: {len(pages)} pages, {n} new ok")
        time.sleep(0.5)
    out = ROOT / "_cand" / f"candidates_{cat}.json"
    out.write_text(json.dumps(list(seen.values()), indent=1))
    print(cat, len(seen), "->", out)


if __name__ == "__main__":
    main()
