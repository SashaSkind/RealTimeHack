"""Stray Scanner export -> depth sidecar -> real-time replay with metric depth."""
import cv2
import numpy as np

from bench.stray_to_sidecar import convert
from handguide.sources import VideoSource


def test_stray_folder_replays_with_lidar_depth(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(30):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    (tmp_path / "depth").mkdir()
    (tmp_path / "confidence").mkdir()
    for i in range(15):                                   # half as many depth frames as video frames
        d = np.full((192, 256), 800 + 10 * i, np.uint16)  # millimetres
        d[0, 0] = 5000
        c = np.full((192, 256), 2, np.uint8)
        c[0, 0] = 0                                       # low confidence -> dropped
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), d)
        cv2.imwrite(str(tmp_path / "confidence" / f"{i:06d}.png"), c)
    convert(tmp_path)
    frames = list(VideoSource(tmp_path / "rgb.mp4", paced=False))
    assert len(frames) == 30 and all(f.depth is not None for f in frames)
    assert abs(float(frames[0].depth[100, 100]) - 0.80) < 1e-3
    assert float(frames[-1].depth[100, 100]) > float(frames[0].depth[100, 100])
    assert frames[0].depth[0, 0] == 0.0
