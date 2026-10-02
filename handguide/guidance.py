"""Hand + target + depth -> cue parameters.

Conventions (read these before touching the math):
- Image coordinates: x grows to the right, y grows DOWNWARD.
- Cue coordinates: dx > 0 means "move right", dy > 0 means "move UP".
  So dy = (hand_y - target_y) / (h/2): a target above the hand has a smaller y,
  which makes dy positive.
- depth_offset > 0 means "reach forward / away from the camera"
  (the target is farther from the camera than the hand).
- distance is 0 when on target and 1 when far away.

States:
- searching: no target known (detector / Cosmos has nothing yet).
- lost:      target known, but the hand is not visible.
- guiding:   both visible, still some way to go.
- close:     within `close_dist`.
- touching:  hand point inside the (slightly expanded) target box AND depth agrees
             (when depth is available).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

SEARCHING, GUIDING, CLOSE, TOUCHING, LOST = "searching", "guiding", "close", "touching", "lost"
STATES = (SEARCHING, GUIDING, CLOSE, TOUCHING, LOST)


@dataclass
class Cue:
    dx: float = 0.0            # -1 (move left) .. +1 (move right)
    dy: float = 0.0            # -1 (move down) .. +1 (move up)
    distance: float = 1.0      # 0 (on target) .. 1 (far)
    state: str = SEARCHING
    depth_offset: float = 0.0  # -1 (pull back) .. +1 (reach forward); 0 if unknown
    depth_kind: str = "none"   # "metric", "relative" or "none"

    def as_dict(self):
        return asdict(self)


@dataclass
class GuidanceConfig:
    dead_zone: float = 0.06        # fraction of the half-frame; offsets inside are zeroed
    box_margin: float = 0.15       # expand target box by this fraction for "inside" tests
    close_dist: float = 0.15
    metric_range_m: float = 0.5    # depth difference that counts as "far" (distance 1)
    metric_touch_m: float = 0.06   # |hand - target| depth to accept as touching
    relative_touch: float = 0.08   # relative-depth equivalent (fraction of target value)
    relative_weight: float = 0.5   # relative depth is noisy; weight it less in distance


def _dead(v: float, dz: float) -> float:
    """Continuous dead zone: 0 inside +/-dz, then ramps linearly to +/-1."""
    a = abs(v)
    if a <= dz:
        return 0.0
    return math.copysign(min(1.0, (a - dz) / (1.0 - dz)), v)


def _clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def compute_cue(
    hand_xy,
    target_box,
    frame_wh,
    hand_depth=None,
    target_depth=None,
    depth_kind: str = "none",
    cfg: GuidanceConfig | None = None,
) -> Cue:
    """Pure function. hand_xy: (x, y) px or None. target_box: (x1, y1, x2, y2) px or None.

    depth_kind:
      "metric":   depths in metres from the camera (LiDAR). Larger = farther.
      "relative": Depth Anything output, larger = CLOSER (inverse depth / disparity).
      "none":     ignore depths.
    """
    cfg = cfg or GuidanceConfig()
    if target_box is None:
        return Cue(state=SEARCHING, depth_kind=depth_kind)
    if hand_xy is None:
        return Cue(state=LOST, depth_kind=depth_kind)

    w, h = frame_wh
    hx, hy = hand_xy
    tx, ty = box_center(target_box)

    raw_dx = _clip((tx - hx) / (0.5 * w))
    raw_dy = _clip((hy - ty) / (0.5 * h))  # image y grows down -> flip so up is positive

    # Hand inside the expanded target box counts as aligned on that axis.
    x1, y1, x2, y2 = target_box
    mx, my = cfg.box_margin * (x2 - x1), cfg.box_margin * (y2 - y1)
    in_x = (x1 - mx) <= hx <= (x2 + mx)
    in_y = (y1 - my) <= hy <= (y2 + my)

    dx = 0.0 if in_x else _dead(raw_dx, cfg.dead_zone)
    dy = 0.0 if in_y else _dead(raw_dy, cfg.dead_zone)

    # Depth term.
    depth_offset = 0.0
    depth_term = 0.0
    depth_ok = None  # None = unknown, True/False = depth agrees / disagrees
    if depth_kind == "metric" and hand_depth and target_depth and hand_depth > 0 and target_depth > 0:
        diff = target_depth - hand_depth  # >0: target farther than hand -> reach forward
        depth_offset = _clip(diff / cfg.metric_range_m)
        depth_term = abs(depth_offset)
        depth_ok = abs(diff) <= cfg.metric_touch_m
    elif depth_kind == "relative" and hand_depth and target_depth and hand_depth > 0 and target_depth > 0:
        # Inverse depth: hand value larger than target value -> hand is closer to the camera
        # than the target -> reach forward.
        rel = (hand_depth - target_depth) / max(target_depth, 1e-6)
        depth_offset = _clip(rel)
        depth_term = cfg.relative_weight * abs(depth_offset)
        depth_ok = abs(rel) <= cfg.relative_touch

    # Unclipped, normalised by frame size: 1 = a full frame width/height away. Using the
    # clipped half-frame offsets here saturated distance at half a frame (caught by Test A).
    planar = math.hypot((tx - hx) / w, (ty - hy) / h)
    distance = _clip(math.hypot(planar, depth_term), 0.0, 1.0)
    if in_x and in_y:
        distance = min(distance, depth_term)

    if in_x and in_y and depth_ok is not False:
        state = TOUCHING
        dx = dy = 0.0
        distance = 0.0 if depth_ok else distance
    elif distance <= cfg.close_dist:
        state = CLOSE
    else:
        state = GUIDING

    return Cue(dx=dx, dy=dy, distance=distance, state=state,
               depth_offset=depth_offset, depth_kind=depth_kind)


@dataclass
class Guide:
    """Stateful wrapper: EMA smoothing + state hysteresis so the audio doesn't flicker.

    - A missing hand holds the last cue for `lost_after` frames before reporting LOST.
    - TOUCHING must hold for `touch_frames` consecutive frames before it is reported.
    """
    cfg: GuidanceConfig = field(default_factory=GuidanceConfig)
    alpha: float = 0.5
    lost_after: int = 5
    touch_frames: int = 3
    _last: Cue | None = None
    _miss: int = 0
    _touch: int = 0

    def update(self, hand_xy, target_box, frame_wh, hand_depth=None, target_depth=None,
               depth_kind="none") -> Cue:
        raw = compute_cue(hand_xy, target_box, frame_wh, hand_depth, target_depth, depth_kind, self.cfg)

        if raw.state == LOST:
            self._miss += 1
            if self._last is not None and self._last.state not in (SEARCHING, LOST) and self._miss < self.lost_after:
                return self._last
            self._last = raw
            self._touch = 0
            return raw
        self._miss = 0

        if raw.state == SEARCHING:
            self._last = raw
            self._touch = 0
            return raw

        self._touch = self._touch + 1 if raw.state == TOUCHING else 0
        state = raw.state
        if state == TOUCHING and self._touch < self.touch_frames:
            state = CLOSE

        prev = self._last if (self._last and self._last.state not in (SEARCHING, LOST)) else None
        if prev is None:
            out = Cue(raw.dx, raw.dy, raw.distance, state, raw.depth_offset, raw.depth_kind)
        else:
            a = self.alpha
            out = Cue(
                dx=a * raw.dx + (1 - a) * prev.dx,
                dy=a * raw.dy + (1 - a) * prev.dy,
                distance=a * raw.distance + (1 - a) * prev.distance,
                state=state,
                depth_offset=a * raw.depth_offset + (1 - a) * prev.depth_offset,
                depth_kind=raw.depth_kind,
            )
            if state == TOUCHING:
                out.dx = out.dy = 0.0
        self._last = out
        return out
