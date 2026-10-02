"""TargetTracker logic with a fake detector: detect-every-N, tracking between detections,
external (Cosmos) box reset with stale-frame compensation, and gating of detections."""
import numpy as np

from handguide.target import Detection, TargetTracker, iou


def scene(cx, cy, size=40, W=320, H=240, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((H, W, 3)) * 60).astype(np.uint8)  # textured background
    patch = np.zeros((size, size, 3), np.uint8)
    patch[::4, :, 0] = 255
    patch[:, ::4, 1] = 255
    patch[size // 4: 3 * size // 4, size // 4: 3 * size // 4, 2] = 255
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    img[y0:y0 + size, x0:x0 + size] = patch
    return img


class FakeDetector:
    def __init__(self, boxes_fn):
        self.boxes_fn = boxes_fn
        self.calls = 0
        self.names = []

    def set_prompts(self, names):
        self.names = names

    def detect(self, rgb):
        self.calls += 1
        return [Detection(b, s, "x") for b, s in self.boxes_fn(self.calls)]


def test_detects_every_n_and_tracks_between():
    det = FakeDetector(lambda i: [((100, 80, 140, 120), 0.9)])
    tt = TargetTracker(det, target="cup", detect_every=5, track_scale=1.0)
    ops = []
    for f in range(10):
        s = tt.update(scene(120 + f, 100))
        ops.append(s.op)
        assert s.box is not None
    assert det.calls == 2
    assert ops[0] == "detect" and ops[5] == "detect" and ops[1:5] == ["track"] * 4


def test_tracker_follows_motion():
    det = FakeDetector(lambda i: [((100, 80, 140, 120), 0.9)] if i == 1 else [])
    tt = TargetTracker(det, target="cup", detect_every=100, track_scale=1.0)
    tt.update(scene(120, 100))
    for f in range(1, 15):
        s = tt.update(scene(120 + 3 * f, 100))
    cx = (s.box[0] + s.box[2]) / 2
    assert abs(cx - (120 + 3 * 14)) < 8, cx


def test_external_box_resets_and_compensates_for_stale_frame():
    det = FakeDetector(lambda i: [])
    tt = TargetTracker(det, target="door handle", detect_every=100, track_scale=1.0)
    old = scene(100, 100)          # frame Cosmos saw
    now = scene(130, 110)          # object moved while the call was in flight
    tt.update(old)
    tt.set_external_box((80, 80, 120, 120), frame_rgb=old)
    s = tt.update(now)
    assert s.source == "external"
    assert iou(s.box, (110, 90, 150, 130)) > 0.5, s.box


def test_detection_far_from_cosmos_box_is_ignored():
    wrong = ((250, 180, 300, 230), 0.95)
    det = FakeDetector(lambda i: [wrong])
    tt = TargetTracker(det, target="door handle", detect_every=1, track_scale=1.0)
    img = scene(100, 100)
    tt.set_external_box((80, 80, 120, 120), frame_rgb=img)
    tt.update(img)
    for _ in range(5):
        s = tt.update(img)
    assert s.box is not None and iou(s.box, (80, 80, 120, 120)) > 0.5, s.box


def test_detection_overlapping_cosmos_box_refines_it():
    det = FakeDetector(lambda i: [((85, 82, 118, 121), 0.6)])
    tt = TargetTracker(det, target="door handle", detect_every=1, track_scale=1.0)
    img = scene(100, 100)
    tt.set_external_box((75, 75, 125, 125), frame_rgb=img)
    tt.update(img)
    s = tt.update(img)
    assert s.source == "detect" and s.box == (85, 82, 118, 121)


def test_target_lost_after_misses_without_external():
    det = FakeDetector(lambda i: [((100, 80, 140, 120), 0.9)] if i == 1 else [])
    tt = TargetTracker(det, target="cup", detect_every=1, max_misses=3, track_scale=1.0, search_every=1)
    blank = np.zeros((240, 320, 3), np.uint8)
    tt.update(scene(120, 100))
    states = [tt.update(blank).box for _ in range(4)]
    assert states[-1] is None


def test_comma_separated_targets_union_prompts():
    det = FakeDetector(lambda i: [])
    tt = TargetTracker(det, target="door handle,door knob", detect_every=1)
    tt.update(np.zeros((240, 320, 3), np.uint8))
    assert "handle" in det.names and "round knob" in det.names


def test_search_every_skips_detection_when_no_target():
    det = FakeDetector(lambda i: [])
    tt = TargetTracker(det, target="cup", detect_every=5, search_every=3)
    for _ in range(9):
        tt.update(np.zeros((240, 320, 3), np.uint8))
    assert det.calls == 3


def test_hold_keeps_box_while_hand_occludes_target():
    det = FakeDetector(lambda i: [((100, 80, 140, 120), 0.9)] if i == 1 else [((250, 180, 290, 220), 0.99)])
    tt = TargetTracker(det, target="cup", detect_every=1, track_scale=1.0)
    img = scene(120, 100)
    first = tt.update(img).box
    for _ in range(5):
        s = tt.update(img, hold=True)
    assert s.op == "hold" and s.box == first and det.calls == 1
