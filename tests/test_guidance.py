"""Test A: guidance math correctness with synthetic positions.

Frame is 640x480. Image y grows DOWN; cue dy > 0 means "move UP".
"""
import pytest

from handguide.guidance import (
    CLOSE, GUIDING, LOST, SEARCHING, TOUCHING, Cue, Guide, GuidanceConfig, compute_cue,
)

W, H = 640, 480
WH = (W, H)


def box_at(cx, cy, half=20):
    return (cx - half, cy - half, cx + half, cy + half)


# --- direction -------------------------------------------------------------

def test_target_right_of_hand_gives_right_cue():
    c = compute_cue((100, 240), box_at(500, 240), WH)
    assert c.dx > 0.5 and c.dy == 0.0 and c.state == GUIDING


def test_target_left_of_hand_gives_left_cue():
    c = compute_cue((500, 240), box_at(100, 240), WH)
    assert c.dx < -0.5 and c.dy == 0.0 and c.state == GUIDING


def test_target_above_hand_gives_up_cue():
    # Target at y=60 (top of image), hand at y=420 (bottom). Target is ABOVE the hand.
    c = compute_cue((320, 420), box_at(320, 60), WH)
    assert c.dy > 0.5, f"target above hand must give UP (dy>0), got dy={c.dy}"
    assert c.dx == 0.0


def test_target_below_hand_gives_down_cue():
    c = compute_cue((320, 60), box_at(320, 420), WH)
    assert c.dy < -0.5, f"target below hand must give DOWN (dy<0), got dy={c.dy}"
    assert c.dx == 0.0


def test_diagonal_up_left():
    c = compute_cue((500, 400), box_at(150, 100), WH)
    assert c.dx < 0 and c.dy > 0


def test_offsets_are_clipped_to_unit_range():
    c = compute_cue((0, 479), box_at(639, 0, half=1), WH)
    assert -1.0 <= c.dx <= 1.0 and -1.0 <= c.dy <= 1.0
    assert c.dx == 1.0 and c.dy == 1.0


# --- distance ----------------------------------------------------------------

def test_near_is_smaller_distance_than_far():
    far = compute_cue((50, 50), box_at(590, 430), WH)
    near = compute_cue((300, 220), box_at(340, 250, half=10), WH)
    assert far.distance > near.distance
    assert far.distance >= 0.9
    assert near.state in (CLOSE, GUIDING)


def test_near_is_close_state():
    c = compute_cue((330, 240), box_at(345, 240, half=5), WH)
    assert c.state == CLOSE
    assert c.distance <= 0.15


def test_distance_monotonic_as_hand_approaches():
    target = box_at(500, 240, half=10)
    ds = [compute_cue((x, 240), target, WH).distance for x in (50, 150, 250, 350, 450)]
    assert all(a > b for a, b in zip(ds, ds[1:])), ds


# --- dead zone / on target -----------------------------------------------------

def test_dead_zone_zeroes_small_offsets():
    cfg = GuidanceConfig(dead_zone=0.06, box_margin=0.0)
    # 10 px offset of a 320 px half-width = 0.031 < dead zone
    c = compute_cue((310, 240), box_at(320, 240, half=2), WH, cfg=cfg)
    assert c.dx == 0.0


def test_dead_zone_is_continuous():
    cfg = GuidanceConfig(dead_zone=0.06, box_margin=0.0)
    # Just outside the dead zone: small but non-zero, no jump to a large value.
    c = compute_cue((320 - 21, 240), box_at(320, 240, half=1), WH, cfg=cfg)
    assert 0.0 < c.dx < 0.02


def test_on_target_is_touching_without_depth():
    c = compute_cue((320, 240), box_at(322, 238), WH)
    assert c.state == TOUCHING
    assert c.dx == 0.0 and c.dy == 0.0 and c.distance == 0.0


def test_on_target_2d_but_wrong_metric_depth_is_not_touching():
    # Hand is 30 cm in front of the target (closer to the camera).
    c = compute_cue((320, 240), box_at(320, 240), WH, hand_depth=0.5, target_depth=0.8, depth_kind="metric")
    assert c.state != TOUCHING
    assert c.depth_offset > 0, "target farther than hand -> reach forward"


def test_on_target_with_matching_metric_depth_is_touching():
    c = compute_cue((320, 240), box_at(320, 240), WH, hand_depth=0.78, target_depth=0.80, depth_kind="metric")
    assert c.state == TOUCHING and c.distance == 0.0


def test_metric_depth_sign_pull_back():
    c = compute_cue((100, 240), box_at(500, 240), WH, hand_depth=0.9, target_depth=0.6, depth_kind="metric")
    assert c.depth_offset < 0, "hand beyond target -> pull back"


def test_relative_depth_sign():
    # Relative depth is inverse depth: larger = closer. Hand closer than target -> reach forward.
    c = compute_cue((100, 240), box_at(500, 240), WH, hand_depth=2.0, target_depth=1.0, depth_kind="relative")
    assert c.depth_offset > 0


def test_missing_depth_values_ignored():
    c = compute_cue((100, 240), box_at(500, 240), WH, hand_depth=0.0, target_depth=None, depth_kind="metric")
    assert c.depth_offset == 0.0 and c.state == GUIDING


# --- missing inputs -------------------------------------------------------------

def test_hand_missing_is_lost():
    c = compute_cue(None, box_at(320, 240), WH)
    assert c.state == LOST


def test_target_missing_is_searching():
    c = compute_cue((320, 240), None, WH)
    assert c.state == SEARCHING


def test_both_missing_is_searching():
    assert compute_cue(None, None, WH).state == SEARCHING


# --- stateful wrapper -----------------------------------------------------------

def test_guide_holds_last_cue_briefly_when_hand_drops():
    g = Guide(lost_after=3, alpha=1.0)
    first = g.update((100, 240), box_at(500, 240), WH)
    assert first.state == GUIDING
    assert g.update(None, box_at(500, 240), WH).state == GUIDING
    assert g.update(None, box_at(500, 240), WH).state == GUIDING
    assert g.update(None, box_at(500, 240), WH).state == LOST


def test_guide_touching_requires_consecutive_frames():
    g = Guide(touch_frames=3, alpha=1.0)
    states = [g.update((320, 240), box_at(320, 240), WH).state for _ in range(4)]
    assert states == [CLOSE, CLOSE, TOUCHING, TOUCHING]


def test_guide_smoothing_keeps_sign():
    g = Guide(alpha=0.5)
    for _ in range(5):
        c = g.update((320, 420), box_at(320, 60), WH)
    assert c.dy > 0.5
