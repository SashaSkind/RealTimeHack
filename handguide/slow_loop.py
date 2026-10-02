"""Slow loop: every 1-2 s send the newest frame to Cosmos Reason (any OpenAI-compatible
endpoint: vLLM on CoreWeave, llama-server locally) and get back strict JSON:

  {"target": str, "affordance": str, "instruction": str,
   "box": [x1,y1,x2,y2] | null, "phase": "searching"|"approaching"|"touching"|"done"}

Runs in its own thread. The fast loop only ever reads `latest()` / receives callbacks;
it never waits. Failures and timeouts keep the last good state.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

PHASES = ("searching", "approaching", "touching", "done")

SYSTEM_PROMPT = (
    "You are the vision module of an assistive device that guides a blind person's hand to an "
    "object. Reply with exactly one JSON object and nothing else: no reasoning, no markdown."
)

# One-line domain hint. Without it, 8B described 7/10 lever handles as "turn" / "turn clockwise"
# (Test C). A terser version ("lever handle: press down; ...") did NOT fix levers, so keep this one.
AFFORDANCE_HINTS = (
    " How common objects are operated: lever door handle - press the lever down, then push or pull "
    "the door; round door knob - grip and turn; cabinet or drawer handle - pull; light switch - "
    "flip or press; cup or mug - grip the handle or body and lift."
)

USER_PROMPT = (
    "Goal: {goal}.\n"
    "Find the object in this image that the user needs for this goal. Reply with JSON only:\n"
    '{{"target": "<short name of the object>", '
    '"box": {box_spec}, '
    '"affordance": "<object type and how to operate it, a few words, e.g. lever handle, press down>", '
    '"instruction": "<one short sentence to say to the user, at most 12 words>", '
    '"phase": "<searching|approaching|touching|done>"}}'
)
# box comes before the free-text fields so it survives if a reply runs away and gets cut off

BOX_SPECS = {
    "norm1000": "[x1, y1, x2, y2] bounding box of the target with coordinates normalized to 0-1000, or null if it is not visible",
    "pixel": "[x1, y1, x2, y2] bounding box of the target in pixel coordinates of this {w}x{h} image, or null if it is not visible",
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength stops the runaway instructions seen in Test C (one reply hit max_tokens)
        "target": {"type": "string", "maxLength": 40},
        "box": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                          {"type": "null"}]},
        "affordance": {"type": "string", "maxLength": 60},
        "instruction": {"type": "string", "maxLength": 90},
        "phase": {"type": "string", "enum": list(PHASES)},
    },
    "required": ["target", "box", "affordance", "instruction", "phase"],
    "additionalProperties": False,
}


@dataclass
class SlowResult:
    ok: bool
    target: str = ""
    affordance: str = ""
    instruction: str = ""
    box: tuple | None = None       # pixel coords in the ORIGINAL frame
    box_raw: list | None = None    # as returned by the model
    phase: str = "searching"
    latency_s: float = 0.0
    frame_ts: float = 0.0
    sent_at: float = 0.0
    error: str = ""
    raw: str = ""
    frame_rgb: np.ndarray | None = field(default=None, repr=False)


def extract_json(text: str) -> dict:
    """Tolerant JSON extraction: strips <think> blocks and code fences, takes the outermost {...}."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"^.*?</think>", "", t, flags=re.S)  # unmatched closing tag
    t = t.replace("```json", "").replace("```", "").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    return json.loads(t[a:b + 1])


def salvage(text: str) -> dict:
    """Last resort for a truncated reply (e.g. hit max_tokens mid-string): pull the fields out
    with regexes and keep only the first sentence of the instruction."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    d = {}
    for k in ("target", "affordance", "instruction", "phase"):
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)', t)
        if m:
            d[k] = m.group(1)
    if "target" not in d:
        raise ValueError("no JSON object in reply")
    if "instruction" in d:
        d["instruction"] = re.split(r"(?<=[.!?])\s", d["instruction"].strip())[0][:90]
    m = re.search(r'"box"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', t)
    d["box"] = [float(v) for v in m.groups()] if m else None
    return d


def validate(d: dict) -> dict:
    out = {}
    for k in ("target", "affordance", "instruction"):
        v = d.get(k, "")
        if not isinstance(v, str):
            raise ValueError(f"{k} is not a string")
        out[k] = v.strip()
    if len(out["instruction"]) >= 85:  # clipped by maxLength: end at the last full word
        out["instruction"] = out["instruction"][:out["instruction"].rfind(" ")].rstrip(",;") or out["instruction"]
    box = d.get("box")
    if box is not None:
        if isinstance(box, dict):
            box = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(isinstance(v, (int, float)) for v in box)):
            raise ValueError(f"bad box {box!r}")
        box = [float(v) for v in box]
    out["box"] = box
    ph = d.get("phase", "searching")
    out["phase"] = ph if ph in PHASES else "searching"
    return out


def box_to_pixels(box, fmt: str, sent_wh, orig_wh):
    """Convert a model box to pixel coords in the original frame. Returns None if unusable."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    ow, oh = orig_wh
    if fmt == "norm1000":
        sx, sy = ow / 1000.0, oh / 1000.0
    elif fmt == "norm1":
        sx, sy = ow, oh
    else:  # pixel coords of the image we sent
        sx, sy = ow / sent_wh[0], oh / sent_wh[1]
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    x1, x2 = max(0.0, x1), min(float(ow), x2)
    y1, y2 = max(0.0, y1), min(float(oh), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def encode_jpeg(rgb: np.ndarray, max_side: int = 640, quality: int = 85):
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = rgb if s == 1.0 else cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode(), (img.shape[1], img.shape[0])


class CosmosClient:
    def __init__(self, url: str, model: str = "cosmos", api_key: str = "none", timeout_s: float = 20.0,
                 box_format: str = "norm1000", json_mode: str = "schema", max_side: int = 640,
                 max_tokens: int = 200, hints: bool = True):
        self.url = url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model, self.api_key, self.timeout_s = model, api_key, timeout_s
        self.box_format, self.json_mode, self.max_side, self.max_tokens = box_format, json_mode, max_side, max_tokens
        self.hints = hints
        self.session = requests.Session()

    def ask(self, rgb: np.ndarray, goal: str) -> SlowResult:
        t0 = time.time()
        b64, sent_wh = encode_jpeg(rgb, self.max_side)
        spec = BOX_SPECS[self.box_format if self.box_format in BOX_SPECS else "norm1000"].format(w=sent_wh[0], h=sent_wh[1])
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + (AFFORDANCE_HINTS if self.hints else "")},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT.format(goal=goal, box_spec=spec)},
                ]},
            ],
        }
        if self.json_mode == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "guidance", "schema": JSON_SCHEMA, "strict": True}}
        elif self.json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        raw = ""
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout_s,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            raw = r.json()["choices"][0]["message"].get("content") or ""
            try:
                d = validate(extract_json(raw))
            except (ValueError, json.JSONDecodeError):
                d = validate(salvage(raw))
            h, w = rgb.shape[:2]
            return SlowResult(ok=True, target=d["target"], affordance=d["affordance"], instruction=d["instruction"],
                              box=box_to_pixels(d["box"], self.box_format, sent_wh, (w, h)), box_raw=d["box"],
                              phase=d["phase"], latency_s=time.time() - t0, raw=raw, sent_at=t0)
        except Exception as e:
            return SlowResult(ok=False, error=f"{type(e).__name__}: {e}"[:400], latency_s=time.time() - t0, raw=raw, sent_at=t0)


class LatestFrame:
    """Single-slot mailbox: the fast loop overwrites, the slow loop reads the newest."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


class SlowLoop(threading.Thread):
    """on_result(result) is called from this thread for every successful call whose content
    is new. It must be thread-safe (TargetTracker.set_external_box and Speaker.speak are)."""

    def __init__(self, client: CosmosClient, frames: LatestFrame, goal: str, interval_s: float = 1.5,
                 on_result=None, log=print):
        super().__init__(daemon=True, name="slow-loop")
        self.client, self.frames, self.goal, self.interval_s = client, frames, goal, interval_s
        self.on_result = on_result
        self.log = log
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_good: SlowResult | None = None
        self.history: list[SlowResult] = []
        self.calls = 0
        self.failures = 0

    def latest(self) -> SlowResult | None:
        with self._lock:
            return self.last_good

    def stop(self):
        self._stop.set()

    def run(self):
        last_ts = None
        while not self._stop.is_set():
            t0 = time.time()
            fr = self.frames.get()
            if fr is None or fr.ts == last_ts:
                self._stop.wait(0.05)
                continue
            last_ts = fr.ts
            res = self.client.ask(fr.rgb, self.goal)
            res.frame_ts = fr.ts
            self.calls += 1
            self.history.append(SlowResult(**{**res.__dict__, "frame_rgb": None}))
            if res.ok:
                res.frame_rgb = fr.rgb
                with self._lock:
                    self.last_good = res
                if self.on_result:
                    try:
                        self.on_result(res)
                    except Exception as e:  # never let a callback kill the thread
                        self.log(f"[slow] on_result error: {e}")
            else:
                self.failures += 1
                self.log(f"[slow] call failed after {res.latency_s:.1f}s: {res.error}")
            wait = self.interval_s - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)
