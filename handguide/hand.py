"""MediaPipe hand landmarker -> palm center (x, y) + confidence.

Palm center = mean of wrist (0) and the four finger MCP joints (5, 9, 13, 17).
It is steadier than a fingertip and is a big enough blob for low-res LiDAR to see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PALM_IDS = (0, 5, 9, 13, 17)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass
class HandResult:
    found: bool = False
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0
    palm_size: float = 0.0          # wrist -> middle MCP distance in px
    handedness: str = ""
    landmarks: np.ndarray | None = None  # (21, 2) px
    all_hands: list = field(default_factory=list)

    @property
    def xy(self):
        return (self.x, self.y) if self.found else None


class HandTracker:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, num_hands: int = 2,
                 mode: str = "video", min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),  # GPU path crashes in mediapipe 1.0.1 on macOS
            running_mode=running,
            num_hands=num_hands,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.mode = mode
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, ts_ms: int | None = None, prefer_near=None) -> HandResult:
        """prefer_near: optional (x, y). With two hands, pick the one nearest this point
        (the target), else the most confident one."""
        h, w = rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            ts = int(ts_ms if ts_ms is not None else self._last_ts + 33)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            res = self.landmarker.detect_for_video(img, ts)
        else:
            res = self.landmarker.detect(img)

        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            palm = pts[list(PALM_IDS)].mean(axis=0)
            hands.append(HandResult(
                found=True, x=float(palm[0]), y=float(palm[1]),
                confidence=float(hd[0].score) if hd else 0.0,
                palm_size=float(np.linalg.norm(pts[0] - pts[9])),
                handedness=hd[0].category_name if hd else "",
                landmarks=pts,
            ))
        if not hands:
            return HandResult()
        if prefer_near is not None and len(hands) > 1:
            px, py = prefer_near
            best = min(hands, key=lambda r: (r.x - px) ** 2 + (r.y - py) ** 2)
        else:
            best = max(hands, key=lambda r: r.confidence)
        best.all_hands = hands
        return best

    def close(self):
        self.landmarker.close()
