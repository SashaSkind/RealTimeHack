"""Open-vocabulary target detection (YOLO-World / YOLOE via ultralytics) + OpenCV tracking.

- Detect every `detect_every` frames; track with CSRT (or KCF) in between.
- `set_external_box()` lets the slow loop (Cosmos) reset the tracker. It is thread-safe:
  the box is queued and applied at the start of the next fast-loop update, because
  OpenCV trackers and the detector must only be touched from the fast-loop thread.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DETECTOR_WEIGHTS = {
    "yoloworld": "yolov8s-worldv2.pt",
    "yoloworld-m": "yolov8m-worldv2.pt",
    "yoloe": "yoloe-11s-seg.pt",
    "yoloe-m": "yoloe-11m-seg.pt",
    "yoloe26": "yoloe-26s-seg.pt",
}

# Prompt sets per target category. Tuned on 48 Commons photos (bench/prompt_search.py):
# literal names are often worse than generic words ("handle" beats "door handle" for levers,
# "switch" beats "light switch"). Door knobs and cabinet handles stay unreliable with every
# model we tried; for those the Cosmos box + tracker is the primary source.
PROMPTS = {
    "door handle": ["handle", "lever handle"],
    "lever handle": ["handle", "lever handle"],
    "door knob": ["round knob", "door knob"],
    "cabinet handle": ["cabinet handle", "drawer handle"],
    "light switch": ["switch", "switch plate"],
    "cup": ["cup", "mug"],
    "mug": ["cup", "mug"],
    "bottle": ["bottle"],
}


@contextmanager
def _in_models_dir():
    """ultralytics downloads weights and text encoders into the CWD; keep them in models/."""
    old = os.getcwd()
    MODELS_DIR.mkdir(exist_ok=True)
    os.chdir(MODELS_DIR)
    try:
        yield
    finally:
        os.chdir(old)


@dataclass
class Detection:
    box: tuple  # x1, y1, x2, y2 px
    score: float
    label: str


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class OpenVocabDetector:
    def __init__(self, kind: str = "yoloe", device: str | None = None, conf: float = 0.1, imgsz: int = 640):
        from ultralytics import YOLOE, YOLOWorld

        self.kind = kind
        weights = DETECTOR_WEIGHTS.get(kind, kind)
        path = MODELS_DIR / weights
        src = str(path) if path.exists() else weights
        self.is_yoloe = "yoloe" in weights
        with _in_models_dir():
            self.model = YOLOE(src) if self.is_yoloe else YOLOWorld(src)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.names: list[str] = []

    def set_prompts(self, names: list[str]):
        names = list(dict.fromkeys(n.strip() for n in names if n and n.strip()))
        if names == self.names:
            return
        with _in_models_dir():
            if self.is_yoloe:
                self.model.set_classes(names, self.model.get_text_pe(names))
            else:
                self.model.set_classes(names)
        self.names = names

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        # ultralytics expects BGR numpy arrays
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            out.append(Detection(tuple(float(v) for v in b), float(s), self.names[int(c)] if int(c) < len(self.names) else str(int(c))))
        out.sort(key=lambda d: -d.score)
        return out


def make_tracker(kind: str = "csrt"):
    if kind == "csrt":
        return cv2.TrackerCSRT_create()
    if kind == "kcf":
        return cv2.TrackerKCF_create()
    if kind == "mil":
        return cv2.TrackerMIL_create()
    raise ValueError(kind)


def _xywh(box):
    x1, y1, x2, y2 = box
    return (int(round(x1)), int(round(y1)), max(2, int(round(x2 - x1))), max(2, int(round(y2 - y1))))


@dataclass
class TargetState:
    box: tuple | None = None
    label: str = ""
    score: float = 0.0
    source: str = "none"   # "detect", "track", "external", "none"
    op: str = "none"       # what ran this frame: "detect", "track", "none"


class TargetTracker:
    def __init__(self, detector: OpenVocabDetector | None, target: str = "cup", detect_every: int = 5,
                 tracker: str = "csrt", max_misses: int = 3, track_scale: float = 0.5, search_every: int = 2):
        self.detector = detector
        self.detect_every = max(1, int(detect_every))
        self.tracker_kind = tracker
        self.max_misses = max_misses
        # With no target, detecting on every frame costs 50-80 ms/frame (Test D) for no benefit:
        # the user can't be guided yet anyway. Search on every `search_every`-th frame instead.
        self.search_every = max(1, int(search_every))
        self._search_i = 0
        self.track_scale = track_scale  # track on a downscaled frame for speed
        self._tracker = None
        self._frame_i = 0
        self._misses = 0
        self._lock = threading.Lock()
        self._pending_external = None
        self._pending_label = None
        self._from_external = False
        self.external_min_iou = 0.1
        self.state = TargetState()
        self.set_target(target)

    # --- target / prompts --------------------------------------------------------
    def set_target(self, target: str):
        self.target = target
        self._pending_label = target

    def _apply_label(self):
        label, self._pending_label = self._pending_label, None
        if label is None or self.detector is None:
            return
        prompts = []
        for part in label.split(","):  # "door handle,door knob" -> union of both prompt sets
            key = part.lower().strip()
            if not key:
                continue
            p = PROMPTS.get(key)
            if p is None:
                p = [key] + next((v for k, v in PROMPTS.items() if k in key or key in k), [])
            prompts += p
        self.detector.set_prompts(prompts or [label])

    def set_external_box(self, box, label: str | None = None, frame_rgb: np.ndarray | None = None):
        """Called from the slow-loop thread. `frame_rgb` is the frame the box refers to; the
        tracker is initialised on that (older) frame and then updated on the current one, which
        compensates for the 1-2 s the Cosmos call took."""
        with self._lock:
            self._pending_external = (tuple(float(v) for v in box), frame_rgb)
            if label:
                self._pending_label = label

    # --- tracker helpers ---------------------------------------------------------
    def _small(self, rgb):
        if self.track_scale == 1.0:
            return rgb
        return cv2.resize(rgb, None, fx=self.track_scale, fy=self.track_scale, interpolation=cv2.INTER_AREA)

    def _init_tracker(self, rgb, box):
        s = self.track_scale
        self._tracker = make_tracker(self.tracker_kind)
        self._tracker.init(self._small(rgb), _xywh([v * s for v in box]))

    def _track(self, rgb):
        if self._tracker is None:
            return None
        ok, (x, y, w, h) = self._tracker.update(self._small(rgb))
        if not ok or w <= 1 or h <= 1:
            return None
        s = self.track_scale
        return (x / s, y / s, (x + w) / s, (y + h) / s)

    def _pick(self, dets: list[Detection]):
        if not dets:
            return None
        if self.state.box is not None:
            if self._from_external:
                # Cosmos chose this target. The open-vocab detector is weak on door hardware
                # (Test B), so only let a detection refine the box if it overlaps it.
                ov = [d for d in dets if iou(d.box, self.state.box) >= self.external_min_iou]
                return max(ov, key=lambda d: d.score + iou(d.box, self.state.box)) if ov else None
            # Prefer continuity with the current target unless something is much more confident.
            return max(dets, key=lambda d: d.score + 0.5 * iou(d.box, self.state.box))
        return dets[0]

    # --- main per-frame update ---------------------------------------------------
    def update(self, rgb: np.ndarray, hold: bool = False) -> TargetState:
        """hold=True: the hand is over the target. Keep the last box instead of tracking/detecting,
        because trackers tend to latch onto the occluding hand and drag the target along with it."""
        if hold and self.state.box is not None and self._pending_external is None:
            self.state = TargetState(box=self.state.box, label=self.state.label, score=self.state.score,
                                     source=self.state.source, op="hold")
            return self.state
        with self._lock:
            ext, self._pending_external = self._pending_external, None
        if self._pending_label is not None:
            self._apply_label()

        if ext is not None:
            box, old = ext
            if old is not None and old.shape == rgb.shape:
                self._init_tracker(old, box)
                tb = self._track(rgb)
                box = tb or box
            self._init_tracker(rgb, box)
            self._from_external = True
            self.state = TargetState(box=box, label=self.target, score=1.0, source="external", op="track")
            self._frame_i = 1
            self._misses = 0
            return self.state

        if self.detector is not None and self._tracker is None:
            self._search_i += 1
            if (self._search_i - 1) % self.search_every != 0:
                self.state = TargetState(op="none")
                return self.state
        else:
            self._search_i = 0
        run_detect = self.detector is not None and (self._frame_i % self.detect_every == 0 or self._tracker is None)
        self._frame_i += 1
        if run_detect:
            dets = self.detector.detect(rgb)
            d = self._pick(dets)
            if d is not None:
                self._init_tracker(rgb, d.box)
                self._misses = 0
                self.state = TargetState(box=d.box, label=d.label, score=d.score, source="detect", op="detect")
                return self.state
            self._misses += 1
            if self._misses >= self.max_misses and not self._from_external:
                self._tracker = None
                self.state = TargetState(op="detect")
                return self.state
            tb = self._track(rgb)
            self.state = TargetState(box=tb, label=self.state.label, score=self.state.score,
                                     source="track" if tb else "none", op="detect+track")
            return self.state

        tb = self._track(rgb)
        if tb is None:
            self._tracker = None
            self._from_external = False
        self.state = TargetState(box=tb, label=self.state.label, score=self.state.score,
                                 source="track" if tb else "none", op="track")
        return self.state
