"""Frame sources. Every source yields Frame(ts, rgb, depth) where
- ts: seconds (media time for files, wall time for live cameras)
- rgb: HxWx3 uint8 in RGB order
- depth: HxW float32 metres (LiDAR) or None

Sources also expose `.stats` (frames read, dropped, time spent sleeping) so the
fast loop can separate real capture cost from pacing waits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    ts: float
    rgb: np.ndarray
    depth: np.ndarray | None = None
    index: int = 0
    due_wall: float | None = None  # wall-clock time this frame "arrived" (for latency)


@dataclass
class SourceStats:
    read: int = 0
    dropped: int = 0
    sleep_s: float = 0.0
    last_capture_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class SourceError(RuntimeError):
    pass


class FrameSource:
    stats: SourceStats

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------
class WebcamSource(FrameSource):
    """OpenCV webcam. NOTE: opening this triggers the macOS camera permission prompt."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.stats = SourceStats()
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise SourceError(f"Could not open webcam index {index}. Check System Settings > "
                              "Privacy & Security > Camera for your terminal app, and the index.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def __iter__(self):
        i = 0
        while True:
            t = time.perf_counter()
            ok, bgr = self.cap.read()
            now = time.perf_counter()
            if not ok:
                break
            self.stats.read += 1
            self.stats.last_capture_ms = (now - t) * 1000
            yield Frame(ts=now, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index=i, due_wall=now)
            i += 1

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------------------
class VideoSource(FrameSource):
    """Plays a video file.

    paced=True (default): frames are released on the file's own clock. If processing
    falls behind, frames that are already overdue are skipped (grabbed, not decoded),
    the way a live camera drops frames. This is what latency numbers should use.
    paced=False: every frame, as fast as possible (offline analysis only).

    Optional sidecar `<video>.depth.npz` (arrays `depth` [N,h,w] float16/32 metres and
    optionally `ts`) supplies per-frame LiDAR depth. `main.py --record` writes these.
    """

    def __init__(self, path: str | Path, paced: bool = True, loop: bool = False, max_width: int | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise SourceError(f"Video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise SourceError(f"Could not open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 <= self.fps <= 240.0):
            self.fps = 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.paced = paced
        self.loop = loop
        self.max_width = max_width
        self.stats = SourceStats(extra={"fps": self.fps, "n_frames": self.n_frames})
        self.depth = None
        side = self.path.with_suffix(self.path.suffix + ".depth.npz")
        if not side.exists():
            side = self.path.with_suffix(".depth.npz")
        if side.exists():
            self.depth = np.load(side)["depth"]

    def _depth_for(self, idx):
        if self.depth is None or idx >= len(self.depth):
            return None
        return self.depth[idx].astype(np.float32)

    def _resize(self, bgr):
        if self.max_width and bgr.shape[1] > self.max_width:
            s = self.max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (self.max_width, int(round(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA)
        return bgr

    def reset_clock(self):
        """Call after warm-up: makes the next frame due now, so model load time isn't
        counted as dropped frames."""
        self._reset = True

    def __iter__(self):
        idx = 0
        start = time.perf_counter()
        self._reset = False
        while True:
            if self._reset:
                start = time.perf_counter() - idx / self.fps
                self._reset = False
            if self.paced:
                now = time.perf_counter()
                due_idx = int((now - start) * self.fps)
                # Drop overdue frames: grab() without decoding.
                while idx < due_idx:
                    if not self.cap.grab():
                        break
                    idx += 1
                    self.stats.dropped += 1
                due = start + idx / self.fps
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                    self.stats.sleep_s += wait
            else:
                due = time.perf_counter()
            t = time.perf_counter()
            ok, bgr = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    idx, start = 0, time.perf_counter()
                    continue
                break
            rgb = cv2.cvtColor(self._resize(bgr), cv2.COLOR_BGR2RGB)
            self.stats.last_capture_ms = (time.perf_counter() - t) * 1000
            self.stats.read += 1
            yield Frame(ts=idx / self.fps, rgb=rgb, depth=self._depth_for(idx), index=idx, due_wall=due)
            idx += 1

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderSource(FrameSource):
    """A folder of stills, sorted by name. Each image is yielded `repeat` times.
    If `fps` is set, frames are paced at that rate (no dropping)."""

    def __init__(self, path: str | Path, repeat: int = 1, fps: float | None = None, max_width: int | None = None):
        self.path = Path(path)
        self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMG_EXT) if self.path.is_dir() else [self.path]
        if not self.files:
            raise SourceError(f"No images in {self.path}")
        self.repeat, self.fps, self.max_width = repeat, fps, max_width
        self.stats = SourceStats(extra={"n_files": len(self.files)})

    def reset_clock(self):
        self._reset = True

    def __iter__(self):
        i = 0
        start = time.perf_counter()
        self._reset = False
        for p in self.files:
            t = time.perf_counter()
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            if self.max_width and bgr.shape[1] > self.max_width:
                s = self.max_width / bgr.shape[1]
                bgr = cv2.resize(bgr, (self.max_width, int(round(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.stats.last_capture_ms = (time.perf_counter() - t) * 1000
            for _ in range(self.repeat):
                if self._reset:
                    start = time.perf_counter() - i / (self.fps or 30.0)
                    self._reset = False
                due = time.perf_counter()
                if self.fps:
                    due = start + i / self.fps
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                        self.stats.sleep_s += wait
                self.stats.read += 1
                yield Frame(ts=i / (self.fps or 30.0), rgb=rgb, index=i, due_wall=due)
                i += 1
            self.current_file = p


# ---------------------------------------------------------------------------
class Record3DSource(FrameSource):
    """iPhone RGB + LiDAR over USB via the Record3D app (adapted from the record3d demo).

    Needs: Record3D app on the phone, "USB Streaming mode" enabled, phone connected by
    cable and trusted. Raises SourceError with a readable message if anything is missing;
    it never takes the rest of the program down.

    Depth is returned at the phone's native depth resolution (e.g. 192x256), in metres;
    pixels with LiDAR confidence 0 are set to 0 (treated as missing downstream).
    """

    DEVICE_TYPE_TRUEDEPTH = 0
    DEVICE_TYPE_LIDAR = 1

    def __init__(self, device_index: int = 0, timeout_s: float = 5.0):
        self.stats = SourceStats()
        try:
            from record3d import Record3DStream
        except Exception as e:  # pragma: no cover - depends on local install
            raise SourceError(f"record3d package not importable ({e}). Install with: "
                              "brew install cmake && pip install record3d") from e
        import threading
        self._Record3DStream = Record3DStream
        self.event = threading.Event()
        self.timeout_s = timeout_s
        self.stopped = False
        try:
            devs = Record3DStream.get_connected_devices()
        except Exception as e:
            raise SourceError(f"record3d could not list devices: {e}") from e
        if len(devs) <= device_index:
            raise SourceError(
                f"No Record3D device at index {device_index} ({len(devs)} found). Check: phone plugged in "
                "by USB and trusted, Record3D app open, Settings > 'USB Streaming mode' on, and the red "
                "record button pressed to start streaming.")
        dev = devs[device_index]
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stopped
        if not self.session.connect(dev):
            raise SourceError("Record3D: connect() failed. Is another program already connected?")
        self.stats.extra.update({"product_id": dev.product_id, "udid": dev.udid})

    def _on_new_frame(self):
        self.event.set()

    def _on_stopped(self):
        self.stopped = True
        self.event.set()

    def intrinsics(self):
        c = self.session.get_intrinsic_mat()
        return np.array([[c.fx, 0, c.tx], [0, c.fy, c.ty], [0, 0, 1]])

    def __iter__(self):
        i = 0
        while not self.stopped:
            if not self.event.wait(self.timeout_s):
                raise SourceError(f"Record3D: no frame for {self.timeout_s}s. Is streaming started in the app?")
            self.event.clear()
            if self.stopped:
                break
            t = time.perf_counter()
            depth = np.array(self.session.get_depth_frame(), dtype=np.float32, copy=True)
            rgb = np.array(self.session.get_rgb_frame(), copy=True)  # already RGB order
            try:
                conf = np.array(self.session.get_confidence_frame(), copy=False)
                if conf.shape == depth.shape and conf.size:
                    depth[conf == 0] = 0.0
            except Exception:
                pass
            if self.session.get_device_type() == self.DEVICE_TYPE_TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)
            depth[~np.isfinite(depth)] = 0.0
            now = time.perf_counter()
            self.stats.read += 1
            self.stats.last_capture_ms = (now - t) * 1000
            yield Frame(ts=now, rgb=rgb, depth=depth, index=i, due_wall=now)
            i += 1

    def close(self):
        self.stopped = True


class PathOnSource(FrameSource):
    """iPhone RGB + LiDAR via the free "Record3DStream - PathOn" iOS app (USB or Wi-Fi).

    Uses that app's Python SDK (`iphone-sensor-sdk`, installed into the venv from
    github.com/YoshiRi/Record3DStream, subdirectory sdk). Its licence allows personal,
    non-commercial use and forbids redistribution, so it is NOT copied into this repo.
    USB mode tunnels TCP through `iproxy` (brew install libimobiledevice).

    RGB arrives as 1920x1440 BGR (JPEG over the wire); depth as 256x192 float32 metres,
    aligned with RGB. Low-confidence depth pixels (0) are set to 0.
    """

    def __init__(self, host: str | None = None, port: int = 8888, timeout_s: float = 5.0, min_conf: int = 1):
        self.stats = SourceStats()
        try:
            from sdk import IPhoneSensorClient
        except Exception as e:
            raise SourceError(
                f"PathOn SDK not importable ({e}). Install: uv pip install "
                "'git+https://github.com/YoshiRi/Record3DStream#subdirectory=sdk' and then reinstall "
                "opencv-contrib-python (the SDK pulls in opencv-python, which hides CSRT).") from e
        usb = host is None
        if usb:
            # iproxy accepts local connections even with no phone attached, so the SDK would
            # report "Connected". Check that a phone is actually on USB first.
            import shutil
            import subprocess
            if shutil.which("idevice_id") is None:
                raise SourceError("PathOn USB needs libimobiledevice: brew install libimobiledevice")
            ids = subprocess.run(["idevice_id", "-l"], capture_output=True, text=True).stdout.split()
            if not ids:
                raise SourceError("PathOn: no iPhone on USB (`idevice_id -l` is empty). Plug in with a data "
                                  "cable, unlock the phone and tap Trust.")
        self.client = IPhoneSensorClient(host or "localhost", port, timeout=timeout_s, usb=usb)
        try:
            ok = self.client.start()
        except Exception as e:
            ok = False
            err = e
        else:
            err = None
        if not ok:
            how = ("USB: phone plugged in, unlocked and trusted, app open and streaming; "
                   "`idevice_id -l` should list the phone") if usb else f"Wi-Fi: same network, IP {host}"
            raise SourceError(f"PathOn: could not connect ({err or 'see message above'}). Check {how}.")
        self.timeout_s, self.min_conf = timeout_s, min_conf
        self.stats.extra.update({"mode": "usb" if usb else f"wifi {host}"})
        # A TCP connection is not proof of a stream: require one real frame.
        self._first = None
        deadline = time.perf_counter() + timeout_s
        while self._first is None and time.perf_counter() < deadline:
            self._first = self.client.wait_for_frame(timeout=0.5)
        if self._first is None:
            self.close()
            raise SourceError(f"PathOn: connected but no frame within {timeout_s:.0f}s. Open the app on the "
                              "phone and make sure it shows it is streaming.")

    def __iter__(self):
        i = 0
        t0 = time.perf_counter()
        last = time.perf_counter()
        while True:
            fr, self._first = (self._first, None) if self._first is not None else (self.client.wait_for_frame(timeout=1.0), None)
            if fr is None:
                if time.perf_counter() - last > self.timeout_s:
                    raise SourceError(f"PathOn: no frame for {self.timeout_s}s. Is the app still streaming?")
                continue
            last = time.perf_counter()
            t = time.perf_counter()
            rgb = cv2.cvtColor(fr.color, cv2.COLOR_BGR2RGB)
            depth = np.array(fr.depth, dtype=np.float32, copy=True)
            conf = getattr(fr, "confidence", None)
            if conf is not None and conf.shape == depth.shape:
                depth[conf < self.min_conf] = 0.0
            depth[~np.isfinite(depth)] = 0.0
            now = time.perf_counter()
            self.stats.read += 1
            self.stats.last_capture_ms = (now - t) * 1000
            yield Frame(ts=now - t0, rgb=rgb, depth=depth, index=i, due_wall=now)
            i += 1

    def close(self):
        try:
            self.client.stop()
        except Exception:
            pass


def open_source(kind: str, path: str | None = None, paced: bool = True, camera_index: int = 0,
                repeat: int = 1, max_width: int | None = None, loop: bool = False) -> FrameSource:
    if kind == "webcam":
        return WebcamSource(camera_index)
    if kind == "video":
        if not path:
            raise SourceError("--source video needs --path FILE")
        return VideoSource(path, paced=paced, loop=loop, max_width=max_width)
    if kind == "images":
        if not path:
            raise SourceError("--source images needs --path DIR")
        return ImageFolderSource(path, repeat=repeat, fps=30.0 if paced else None, max_width=max_width)
    if kind == "record3d":
        return Record3DSource(camera_index)
    if kind == "pathon":
        return PathOnSource(host=path)  # --path IPHONE_IP for Wi-Fi; omit for USB
    raise SourceError(f"unknown source {kind}")
