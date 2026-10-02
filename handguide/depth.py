"""Depth at the palm and at the target.

- LiDAR frame present -> metric (metres). Median of valid (>0, finite) pixels in a small
  window. The depth map may be lower resolution than RGB (Record3D LiDAR is 192x256), so
  coordinates are scaled.
- No LiDAR -> Depth Anything V2 Small (relative inverse depth, larger = closer). Only use
  it as "closer / farther than the target", never as centimetres.
- Both readings are smoothed with a running median over the last few frames.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def sample_depth(depth: np.ndarray, xy, rgb_shape, radius_frac: float = 0.02, min_radius: int = 2):
    """Median of valid depth in a window around xy (given in RGB pixel coords)."""
    if depth is None or xy is None:
        return None
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[:2]
    sx, sy = dw / rw, dh / rh
    cx, cy = xy[0] * sx, xy[1] * sy
    r = max(min_radius, int(round(radius_frac * max(dw, dh))))
    x0, x1 = int(max(0, cx - r)), int(min(dw, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(dh, cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    win = depth[y0:y1, x0:x1].astype(np.float32).ravel()
    win = win[np.isfinite(win) & (win > 0)]
    if win.size < 3:
        return None
    return float(np.median(win))


def sample_box_depth(depth: np.ndarray, box, rgb_shape, inner: float = 0.5):
    """Median depth over the central `inner` fraction of a box."""
    if depth is None or box is None:
        return None
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw, hh = 0.5 * inner * (x2 - x1), 0.5 * inner * (y2 - y1)
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[:2]
    sx, sy = dw / rw, dh / rh
    xa, xb = int(max(0, (cx - hw) * sx)), int(min(dw, np.ceil((cx + hw) * sx) + 1))
    ya, yb = int(max(0, (cy - hh) * sy)), int(min(dh, np.ceil((cy + hh) * sy) + 1))
    if xb <= xa or yb <= ya:
        return sample_depth(depth, (cx, cy), rgb_shape)
    win = depth[ya:yb, xa:xb].astype(np.float32).ravel()
    win = win[np.isfinite(win) & (win > 0)]
    if win.size < 3:
        return sample_depth(depth, (cx, cy), rgb_shape)
    return float(np.median(win))


class RunningMedian:
    def __init__(self, n: int = 5):
        self.buf = deque(maxlen=n)

    def update(self, v):
        if v is None:
            return float(np.median(self.buf)) if self.buf else None
        self.buf.append(v)
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()


class DepthAnything:
    """Depth Anything V2 Small (Apache-2.0) via transformers, on MPS if available."""

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    CACHE_DIR = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "models" / "hf")  # keep weights in the repo

    def __init__(self, device: str | None = None, input_size: int = 252):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.CACHE_DIR)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID, cache_dir=self.CACHE_DIR).to(self.device).eval()
        self.input_size = input_size  # multiple of 14
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Returns relative inverse depth at a reduced resolution (larger = closer)."""
        torch = self.torch
        import cv2
        h, w = rgb.shape[:2]
        s = self.input_size / max(h, w)
        nh, nw = max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            x = torch.from_numpy(small).to(self.device).permute(2, 0, 1)[None].float() / 255.0
            x = (x - self.mean) / self.std
            out = self.model(pixel_values=x).predicted_depth
            if self.device == "mps":
                torch.mps.synchronize()
        return out[0].float().cpu().numpy()


@dataclass
class DepthReading:
    hand: float | None = None
    target: float | None = None
    kind: str = "none"  # "metric", "relative", "none"
    ran_model: bool = False


class DepthEstimator:
    """mode: "auto" (LiDAR if present else DAv2), "lidar", "dav2", "off"."""

    def __init__(self, mode: str = "auto", every: int = 1, smooth_n: int = 5, device: str | None = None,
                 input_size: int = 252):
        self.mode = mode
        self.every = max(1, every)
        self.dav2 = None
        if mode in ("auto", "dav2"):
            self._dav2_args = (device, input_size)
        self.hand_f = RunningMedian(smooth_n)
        self.target_f = RunningMedian(smooth_n)
        self._i = 0
        self._last_map = None
        self._last_kind = "none"

    def _get_dav2(self):
        if self.dav2 is None:
            self.dav2 = DepthAnything(*self._dav2_args)
        return self.dav2

    def warmup(self, rgb):
        if self.mode == "dav2":
            self._get_dav2()(rgb)

    def update(self, rgb, lidar, hand_xy, target_box, palm_size: float = 0.0,
               freeze_target: bool = False) -> DepthReading:
        """freeze_target=True while the hand is over the target box: the box then samples the
        hand, not the object, and within a few frames target depth would equal hand depth and
        fake a "touching". Keep the last object depth instead."""
        if self.mode == "off":
            return DepthReading()
        ran = False
        if lidar is not None and self.mode in ("auto", "lidar"):
            dmap, kind = lidar, "metric"
        elif self.mode in ("auto", "dav2"):
            if self._last_map is None or self._i % self.every == 0:
                self._last_map = self._get_dav2()(rgb)
                ran = True
            dmap, kind = self._last_map, "relative"
            self._i += 1
        else:
            return DepthReading()

        if kind != self._last_kind:
            self.hand_f.reset()
            self.target_f.reset()
            self._last_kind = kind

        # Palm window ~ a third of the palm size, as a fraction of the RGB frame.
        rfrac = 0.02
        if palm_size > 0:
            rfrac = max(0.01, min(0.05, (palm_size / 3.0) / max(rgb.shape[:2])))
        hd = sample_depth(dmap, hand_xy, rgb.shape, radius_frac=rfrac)
        td = sample_box_depth(dmap, target_box, rgb.shape)
        if hand_xy is None:
            self.hand_f.reset()
        if target_box is None:
            self.target_f.reset()
        if target_box is None:
            tval = None
        elif freeze_target and self.target_f.buf:
            tval = self.target_f.update(None)  # last smoothed object depth, no new sample
        else:
            tval = self.target_f.update(td)
        return DepthReading(hand=self.hand_f.update(hd) if hand_xy is not None else None,
                            target=tval,
                            kind=kind, ran_model=ran)
