"""Frame sources: pacing/dropping, depth sidecar replay, and Record3D failing cleanly."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from handguide.depth import sample_box_depth, sample_depth
from handguide.main import Recorder
from handguide.sources import Frame, ImageFolderSource, Record3DSource, SourceError, VideoSource, open_source


def _make_video(path, n=60, fps=30, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 4 % 255, np.uint8)
        w.write(img)
    w.release()


def test_paced_video_drops_frames_when_processing_is_slow(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, n=60, fps=30)  # 2 s
    src = VideoSource(p, paced=True)
    t = time.perf_counter()
    got = []
    for fr in src:
        got.append(fr.index)
        time.sleep(0.1)  # "processing" at 10 fps
    wall = time.perf_counter() - t
    assert 1.8 < wall < 2.6, wall                     # plays in real time
    assert 15 <= len(got) <= 25, len(got)             # ~1 in 3 frames processed
    assert src.stats.dropped >= 35
    assert got == sorted(got)


def test_paced_video_keeps_real_time_when_fast(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, n=30, fps=30)
    src = VideoSource(p, paced=True)
    t = time.perf_counter()
    n = sum(1 for _ in src)
    assert n == 30 and src.stats.dropped == 0
    assert time.perf_counter() - t > 0.9


def test_unpaced_video_processes_every_frame_fast(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p, n=60, fps=30)
    src = VideoSource(p, paced=False)
    t = time.perf_counter()
    n = sum(1 for _ in src)
    assert n == 60 and src.stats.dropped == 0
    assert time.perf_counter() - t < 1.0


def test_recorder_roundtrip_with_depth(tmp_path):
    rec = Recorder(tmp_path / "rec", fps=30)
    for i in range(30):
        d = np.full((192, 256), 0.5 + i * 0.01, np.float32)
        rec.add(Frame(ts=i / 15.0, rgb=np.full((240, 320, 3), 100, np.uint8), depth=d))  # 15 fps in
    rec.close()
    src = VideoSource(tmp_path / "rec" / "rgb.mp4", paced=False)
    frames = list(src)
    assert 55 <= len(frames) <= 60  # resampled to 30 fps by duplicating
    assert all(f.depth is not None and f.depth.shape == (192, 256) for f in frames)
    assert frames[-1].depth[0, 0] > frames[0].depth[0, 0]


def test_image_folder(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((50, 60, 3), np.uint8))
    frames = list(ImageFolderSource(tmp_path, repeat=2))
    assert len(frames) == 6 and frames[0].rgb.shape == (50, 60, 3)


def test_missing_video_is_a_clear_error(tmp_path):
    with pytest.raises(SourceError):
        open_source("video", str(tmp_path / "nope.mp4"))


def test_record3d_without_phone_raises_source_error():
    with pytest.raises(SourceError) as e:
        Record3DSource(0)
    assert "Record3D" in str(e.value) or "record3d" in str(e.value)


def test_main_exits_cleanly_without_phone():
    from handguide.main import parse_args, run
    assert run(parse_args(["--source", "record3d", "--no-audio"])) == 2


# --- depth sampling ------------------------------------------------------------

def test_depth_sampling_scales_and_ignores_zeros():
    depth = np.full((192, 256), 0.8, np.float32)
    depth[90:100, 120:130] = 0.0          # hole at the palm location
    depth[80:110, 110:140][depth[80:110, 110:140] > 0] = 0.5
    rgb_shape = (720, 960, 3)             # depth is 1/3.75 of RGB
    v = sample_depth(depth, (125 * 3.75, 95 * 3.75), rgb_shape, radius_frac=0.04)
    assert v == pytest.approx(0.5)
    assert sample_depth(np.zeros((192, 256), np.float32), (100, 100), rgb_shape) is None
    d = np.full((192, 256), 1.2, np.float32)
    assert sample_box_depth(d, (0, 0, 960, 720), rgb_shape) == pytest.approx(1.2)
    d[:, :] = np.nan
    assert sample_box_depth(d, (0, 0, 960, 720), rgb_shape) is None


def test_pathon_without_phone_raises_source_error(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from handguide.sources import PathOnSource
    # simulate "no iPhone on USB" regardless of what is plugged in right now
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(SourceError, match="no iPhone"):
        PathOnSource(timeout_s=1.0)


def test_rotated_source_keeps_rgb_and_depth_aligned(tmp_path):
    from handguide.sources import RotatedSource

    class One:
        stats = None

        def __iter__(self):
            rgb = np.zeros((120, 160, 3), np.uint8)
            rgb[0:10, 0:20] = 255                      # marker at top-left
            d = np.zeros((48, 64), np.float32)
            d[0:4, 0:8] = 1.0                          # same marker in depth
            yield Frame(ts=0, rgb=rgb, depth=d)

    fr = next(iter(RotatedSource(One(), 90)))
    assert fr.rgb.shape == (160, 120, 3) and fr.depth.shape == (64, 48)
    # clockwise: top-left goes to top-right, in both images
    assert fr.rgb[0:20, -10:].mean() == 255 and fr.depth[0:8, -4:].mean() == 1.0
