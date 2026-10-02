"""LiDAR: the hand in FRONT of the cup (covering it in the image) must not count as touching."""
import numpy as np

from handguide.depth import DepthEstimator
from handguide.guidance import TOUCHING, compute_cue

RGB = np.zeros((480, 640, 3), np.uint8)
BOX = (300, 220, 340, 260)       # cup in RGB pixels
CENTER = (320, 240)


def lidar(hand_over_cup: bool, hand_depth=0.50, cup_depth=0.80):
    d = np.full((192, 256), 1.5, np.float32)          # wall
    d[88:104, 120:136] = cup_depth                     # cup (RGB box scaled by 0.4)
    if hand_over_cup:
        d[84:108, 116:140] = hand_depth                # hand covers the cup, 30 cm in front
    return d


def run(freeze):
    de = DepthEstimator("lidar")
    for _ in range(5):                                 # cup visible, hand elsewhere
        de.update(RGB, lidar(False), (100, 400), BOX)
    states = []
    for _ in range(8):                                 # hand moves in front of the cup
        r = de.update(RGB, lidar(True), CENTER, BOX, freeze_target=freeze)
        states.append((compute_cue(CENTER, BOX, (640, 480), r.hand, r.target, r.kind).state, r.target))
    return states


def test_without_freeze_occlusion_fakes_touching():
    # Documents the failure mode the freeze fixes.
    assert run(freeze=False)[-1][0] == TOUCHING


def test_freeze_keeps_cup_depth_and_does_not_touch():
    states = run(freeze=True)
    assert all(s != TOUCHING for s, _ in states)
    assert abs(states[-1][1] - 0.80) < 1e-6
