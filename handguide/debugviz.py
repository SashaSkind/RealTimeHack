"""Live "show the maths" panel, drawn next to the camera overlay (--display / --save-video).

It re-derives every intermediate number of guidance.compute_cue from the same inputs, so you
can read the whole chain: palm & target pixels -> raw offsets (with the image-y flip) -> dead
zone -> planar distance -> depth term -> distance -> audio parameters. Below that: a direction
pad, the last 5 s of dx/dy, distance and depth, and per-stage timings.

Colours: dark surface + two validated series hues (blue, orange; dataviz validator, all checks
pass on #1a1a19). State uses status colours but always with its text label.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .audio import AudioConfig, cue_to_params
from .guidance import GuidanceConfig, _dead, box_center

# BGR
SURFACE = (25, 26, 26)        # #1a1a19
GRID = (53, 56, 56)           # #383835
TEXT = (255, 255, 255)
TEXT2 = (183, 194, 195)       # #c3c2b7
BLUE = (229, 135, 57)         # #3987e5  series 1
ORANGE = (38, 89, 217)        # #d95926  series 2
STATE_COLOR = {"touching": (12, 163, 12), "close": (25, 178, 250), "lost": (59, 59, 208),
               "guiding": TEXT2, "searching": GRID}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _t(img, s, x, y, scale=0.45, color=TEXT2, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


class DebugPanel:
    def __init__(self, width: int = 460, seconds: float = 5.0, gcfg: GuidanceConfig | None = None,
                 acfg: AudioConfig | None = None):
        self.w = width
        self.seconds = seconds
        self.gcfg = gcfg or GuidanceConfig()
        self.acfg = acfg or AudioConfig()
        self.hist = deque()

    # ------------------------------------------------------------------------------------
    def render(self, height, frame_wh, t, hand, tstate, cue, depth, timing: dict | None = None,
               fps: float | None = None):
        W, H = frame_wh
        img = np.full((max(height, 700), self.w, 3), SURFACE, np.uint8)
        x0 = 14
        self.hist.append((t, cue.dx, cue.dy, cue.distance, depth.hand, depth.target))
        while self.hist and t - self.hist[0][0] > self.seconds:
            self.hist.popleft()

        # --- header: state (colour + label), fps, latency
        col = STATE_COLOR.get(cue.state, TEXT2)
        cv2.circle(img, (x0 + 8, 24), 8, col, -1, cv2.LINE_AA)
        _t(img, cue.state.upper(), x0 + 24, 31, 0.75, TEXT, 2)
        hdr = []
        if fps:
            hdr.append(f"{fps:.0f} fps")
        if timing and isinstance(timing.get("latency"), float):
            hdr.append(f"frame->cue {timing['latency']:.0f} ms")
        _t(img, "  ".join(hdr), self.w - 190, 30, 0.45, TEXT2)
        tl = f"target: {tstate.label or '-'} ({tstate.source}, {tstate.op})"
        hl = f"palm conf {hand.confidence:.2f}" if hand.found else "palm: not found"
        _t(img, tl, x0, 56)
        _t(img, hl, x0 + 290, 56)

        # --- the maths
        y = 84
        lines = self._math_lines(hand, tstate, cue, depth, W, H)
        for s, c in lines:
            _t(img, s, x0, y, 0.44, c)
            y += 19
        y += 6

        # --- direction pad + distance bar
        pad = 120
        self._pad(img, x0, y, pad, cue)
        self._bar(img, x0 + pad + 30, y + 10, self.w - (x0 + pad + 30) - 16, cue.distance)
        y += pad + 14

        # --- sparklines
        ts = [h[0] for h in self.hist]
        y = self._spark(img, x0, y, "dx / dy  (move right / up = +)", ts,
                        [([h[1] for h in self.hist], BLUE, "dx"), ([h[2] for h in self.hist], ORANGE, "dy")],
                        (-1, 1), zero=True)
        y = self._spark(img, x0, y, "distance (0 = on target)", ts, [([h[3] for h in self.hist], BLUE, None)], (0, 1), fmt="{:.0f}")
        unit = "m" if depth.kind == "metric" else "rel." if depth.kind == "relative" else ""
        dvals = [v for h in self.hist for v in (h[4], h[5]) if v is not None]
        rng = (min(dvals), max(dvals)) if dvals else (0, 1)
        if rng[1] - rng[0] < 0.05:
            rng = (rng[0] - 0.05, rng[1] + 0.05)
        y = self._spark(img, x0, y, f"depth {unit} ({depth.kind})", ts,
                        [([h[4] for h in self.hist], BLUE, "hand"), ([h[5] for h in self.hist], ORANGE, "target")],
                        rng, fmt="{:.2f}")

        # --- stage timings
        if timing:
            parts = [f"{k} {timing[k]:.0f}" for k in ("hand", "detect", "track", "hold", "depth", "guidance")
                     if isinstance(timing.get(k), float)]
            if isinstance(timing.get("total"), float):
                parts.append(f"total {timing['total']:.0f} ms")
            _t(img, "  ".join(parts), x0, img.shape[0] - 14, 0.42, TEXT2)
        return img

    # ------------------------------------------------------------------------------------
    def _math_lines(self, hand, ts, cue, depth, W, H):
        g = self.gcfg
        L = []
        if ts.box is None:
            return [("no target -> state SEARCHING (silent)", TEXT2)]
        tx, ty = box_center(ts.box)
        if not hand.found:
            return [(f"target centre ({tx:.0f}, {ty:.0f}) px", TEXT2),
                    ("no palm -> hold last cue, then LOST (low pulse)", TEXT2)]
        hx, hy = hand.x, hand.y
        ddx, ddy = tx - hx, ty - hy
        rdx = max(-1, min(1, ddx / (0.5 * W)))
        rdy = max(-1, min(1, (hy - ty) / (0.5 * H)))
        x1, y1, x2, y2 = ts.box
        mx, my = g.box_margin * (x2 - x1), g.box_margin * (y2 - y1)
        in_x = (x1 - mx) <= hx <= (x2 + mx)
        in_y = (y1 - my) <= hy <= (y2 + my)
        planar = math.hypot(ddx / W, ddy / H)
        L.append((f"palm ({hx:.0f}, {hy:.0f})   target ({tx:.0f}, {ty:.0f})   frame {W}x{H}", TEXT2))
        L.append((f"dx_raw = ({tx:.0f}-{hx:.0f}) / {W / 2:.0f} = {rdx:+.2f}" + ("   [palm inside box: 0]" if in_x else ""), TEXT))
        L.append((f"dy_raw = ({hy:.0f}-{ty:.0f}) / {H / 2:.0f} = {rdy:+.2f}  (image y is down)" + ("  [inside: 0]" if in_y else ""), TEXT))
        zdx = 0.0 if in_x else _dead(rdx, g.dead_zone)
        zdy = 0.0 if in_y else _dead(rdy, g.dead_zone)
        L.append((f"dead zone {g.dead_zone:.2f} -> {zdx:+.2f}, {zdy:+.2f}   smoothed -> dx {cue.dx:+.2f}  dy {cue.dy:+.2f}", TEXT))
        L.append((f"planar = hypot({ddx:.0f}/{W}, {ddy:.0f}/{H}) = {planar:.2f}", TEXT))
        if depth.kind == "metric" and depth.hand and depth.target:
            diff = depth.target - depth.hand
            L.append((f"depth: hand {depth.hand:.3f} m  target {depth.target:.3f} m  diff {diff * 100:+.1f} cm", TEXT))
            L.append((f"depth_offset = {diff:+.2f}/{g.metric_range_m} = {cue.depth_offset:+.2f}  "
                      f"touch if |diff| <= {g.metric_touch_m * 100:.0f} cm", TEXT2))
        elif depth.kind == "relative" and depth.hand and depth.target:
            rel = (depth.hand - depth.target) / max(depth.target, 1e-6)
            L.append((f"rel. depth (bigger=closer): hand {depth.hand:.2f}  target {depth.target:.2f}", TEXT))
            L.append((f"rel = {rel:+.2f} -> {'reach forward' if rel > 0 else 'pull back'}  (x{g.relative_weight} weight)", TEXT2))
        else:
            L.append(("depth: off / unavailable -> touching = overlap in the image only", TEXT2))
        L.append((f"distance = hypot(planar, depth term) = {cue.distance:.2f}  (smoothed)", TEXT))
        pan, freq, rate, mode = cue_to_params(cue, self.acfg)
        L.append((f"audio: pan {pan:+.2f}  pitch {freq:.0f} Hz  {rate:.1f} beeps/s  [{mode}]", TEXT))
        return L

    def _pad(self, img, x, y, s, cue):
        cv2.rectangle(img, (x, y), (x + s, y + s), GRID, 1)
        cx, cy = x + s // 2, y + s // 2
        cv2.line(img, (x, cy), (x + s, cy), GRID, 1)
        cv2.line(img, (cx, y), (cx, y + s), GRID, 1)
        dz = int(self.gcfg.dead_zone * s / 2)
        cv2.rectangle(img, (cx - dz, cy - dz), (cx + dz, cy + dz), GRID, 1)
        _t(img, "up", cx - 8, y - 4, 0.38)
        _t(img, "down", cx - 15, y + s + 13, 0.38)
        _t(img, "L", x - 11, cy + 4, 0.38)
        _t(img, "R", x + s + 3, cy + 4, 0.38)
        px, py = int(cx + cue.dx * s / 2), int(cy - cue.dy * s / 2)
        cv2.line(img, (cx, cy), (px, py), BLUE, 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 6, BLUE, -1, cv2.LINE_AA)

    def _bar(self, img, x, y, w, d):
        _t(img, f"distance {d:.2f}", x, y + 4, 0.45, TEXT)
        by = y + 16
        cv2.rectangle(img, (x, by), (x + w, by + 10), GRID, -1)
        cv2.rectangle(img, (x, by), (x + int(w * max(0, min(1, d))), by + 10), BLUE, -1)
        g = self.gcfg
        cx = x + int(w * g.close_dist)
        cv2.line(img, (cx, by - 3), (cx, by + 13), TEXT2, 1)
        _t(img, "close", cx - 14, by + 28, 0.36)
        _t(img, "0", x, by + 28, 0.36)
        _t(img, "1", x + w - 6, by + 28, 0.36)

    def _spark(self, img, x, y, title, ts, series, rng, zero=False, fmt="{:+.0f}"):
        w, h = self.w - x - 46, 58
        _t(img, title, x, y + 12, 0.42, TEXT2)
        lx = x + 250
        for _, col, name in series:
            if name:
                cv2.line(img, (lx, y + 8), (lx + 14, y + 8), col, 2)
                _t(img, name, lx + 18, y + 12, 0.4, TEXT2)
                lx += 70
        top = y + 20
        lo, hi = rng
        cv2.rectangle(img, (x, top), (x + w, top + h), GRID, 1)
        _t(img, fmt.format(hi), x + w + 4, top + 10, 0.36)
        _t(img, fmt.format(lo), x + w + 4, top + h, 0.36)
        if zero and lo < 0 < hi:
            zy = int(top + h * (hi - 0) / (hi - lo))
            cv2.line(img, (x, zy), (x + w, zy), GRID, 1)
        if len(ts) > 1:
            t1 = ts[-1]
            for vals, col, _ in series:
                pts = [(int(x + w * (1 - (t1 - t) / self.seconds)), int(top + h * (hi - v) / (hi - lo)))
                       for t, v in zip(ts, vals) if v is not None and lo - 1e-9 <= v <= hi + 1e-9]
                if len(pts) > 1:
                    cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        return top + h + 16
