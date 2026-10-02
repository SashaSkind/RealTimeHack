"""Stereo tone generator driven by cue parameters.

Mapping:
- pan     <- dx        (equal-power pan, -1 = left ear, +1 = right ear)
- pitch   <- dy        (base * 2**(dy * octaves); up = higher)
- beeps/s <- distance  (faster when closer)
- touching -> steady confirmation tone; lost -> slow low pulse; searching -> silence

Click-free: the oscillator phase is accumulated continuously, every parameter is ramped
linearly across each audio block toward a one-pole-smoothed target, and beeps have
fixed-length attack/release edges.

`ToneSynth.render()` is pure numpy so it can be tested and rendered to WAV offline.
`AudioOut` plays it through sounddevice. `CueLogger` is the --no-audio stand-in.
"""
from __future__ import annotations

import csv
import math
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .guidance import CLOSE, GUIDING, LOST, SEARCHING, TOUCHING, Cue


@dataclass
class AudioConfig:
    sr: int = 48000
    base_hz: float = 523.25      # C5 at dy = 0
    octaves: float = 1.0         # dy = +/-1 -> +/- this many octaves
    rate_far: float = 2.0        # beeps per second at distance 1
    rate_near: float = 14.0      # beeps per second at distance 0
    beep_on_s: float = 0.045
    edge_s: float = 0.008        # attack / release
    smooth_s: float = 0.04       # parameter smoothing time constant
    touch_hz: float = 880.0
    lost_hz: float = 220.0
    lost_rate: float = 1.5
    volume: float = 0.25


def cue_to_params(cue: Cue, cfg: AudioConfig):
    """Returns (pan, freq_hz, beep_rate, mode) where mode in beep/steady/silent."""
    if cue.state == SEARCHING:
        return 0.0, cfg.base_hz, cfg.rate_far, "silent"
    if cue.state == LOST:
        return 0.0, cfg.lost_hz, cfg.lost_rate, "beep"
    if cue.state == TOUCHING:
        return 0.0, cfg.touch_hz, cfg.rate_near, "steady"
    pan = max(-1.0, min(1.0, cue.dx))
    freq = cfg.base_hz * 2.0 ** (max(-1.0, min(1.0, cue.dy)) * cfg.octaves)
    d = max(0.0, min(1.0, cue.distance))
    rate = cfg.rate_far + (cfg.rate_near - cfg.rate_far) * (1.0 - d)
    return pan, freq, rate, "beep"


class ToneSynth:
    def __init__(self, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()
        c = self.cfg
        self._target = (0.0, c.base_hz, c.rate_far, "silent")
        self.pan = 0.0
        self.freq = c.base_hz
        self.gain = 0.0          # master gain (0 when silent)
        self.steady = 0.0        # 0 = beeping, 1 = steady tone
        self.phase = 0.0
        self.t_cycle = 0.0
        self.rate = c.rate_far

    def set_params(self, pan, freq, rate, mode):
        self._target = (float(pan), float(freq), float(rate), mode)

    def set_cue(self, cue: Cue):
        self.set_params(*cue_to_params(cue, self.cfg))

    def _approach(self, cur, tgt, n):
        k = 1.0 - math.exp(-n / (self.cfg.smooth_s * self.cfg.sr))
        return cur + (tgt - cur) * k

    def render(self, n: int) -> np.ndarray:
        c = self.cfg
        pan_t, freq_t, rate_t, mode = self._target
        gain_t = 0.0 if mode == "silent" else 1.0
        steady_t = 1.0 if mode == "steady" else 0.0

        ramp = np.arange(1, n + 1, dtype=np.float64) / n
        pan_n = self._approach(self.pan, pan_t, n)
        # smooth pitch in log space so glides sound even
        freq_n = math.exp(self._approach(math.log(self.freq), math.log(freq_t), n))
        gain_n = self._approach(self.gain, gain_t, n)
        steady_n = self._approach(self.steady, steady_t, n)

        pan = self.pan + (pan_n - self.pan) * ramp
        freq = self.freq + (freq_n - self.freq) * ramp
        gain = self.gain + (gain_n - self.gain) * ramp
        steady = self.steady + (steady_n - self.steady) * ramp

        # continuous-phase oscillator
        inc = 2.0 * math.pi * freq / c.sr
        ph = self.phase + np.cumsum(inc)
        tone = np.sin(ph)
        self.phase = float(ph[-1] % (2.0 * math.pi))

        # beep envelope from time-in-cycle with fixed on-length
        self.rate = rate_t
        period = max(1.0 / max(rate_t, 0.1), c.beep_on_s + 2 * c.edge_s)
        if self.t_cycle >= period:  # period shrank; we were in the silent gap, so env was 0
            self.t_cycle = 0.0
        t = (self.t_cycle + np.arange(n) / c.sr) % period
        self.t_cycle = float((self.t_cycle + n / c.sr) % period)
        env = np.clip(t / c.edge_s, 0, 1) * np.clip((c.beep_on_s - t) / c.edge_s, 0, 1)
        env = np.where(t <= c.beep_on_s, env, 0.0)

        amp = c.volume * gain * (env * (1.0 - steady) + steady)
        theta = (pan + 1.0) * (math.pi / 4.0)
        out = np.empty((n, 2), dtype=np.float32)
        out[:, 0] = tone * amp * np.cos(theta)
        out[:, 1] = tone * amp * np.sin(theta)

        self.pan, self.freq, self.gain, self.steady = pan_n, freq_n, gain_n, steady_n
        return out


def write_wav(path: str | Path, stereo: np.ndarray, sr: int):
    data = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


class AudioOut:
    """Real-time playback via sounddevice. Opening the output device does not need any
    macOS permission (only microphone input does)."""

    def __init__(self, cfg: AudioConfig | None = None, blocksize: int = 256):
        import sounddevice as sd

        self.synth = ToneSynth(cfg)
        self.stream = sd.OutputStream(samplerate=self.synth.cfg.sr, channels=2, dtype="float32",
                                      blocksize=blocksize, callback=self._cb, latency="low")
        self.stream.start()

    def _cb(self, outdata, frames, time_info, status):
        outdata[:] = self.synth.render(frames)

    def update(self, cue: Cue):
        self.synth.set_cue(cue)

    def close(self):
        self.synth.set_params(0, self.synth.cfg.base_hz, 1, "silent")
        time.sleep(0.1)
        self.stream.stop()
        self.stream.close()


class CueLogger:
    """--no-audio: records the cue and the audio parameters it would have produced."""

    FIELDS = ["t", "frame", "state", "dx", "dy", "distance", "depth_offset", "depth_kind",
              "hand_depth", "target_depth", "hand_x", "hand_y", "target_box",
              "pan", "freq_hz", "beep_rate", "mode"]

    def __init__(self, path: str | Path | None = None, print_every: int = 0, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()
        self.rows = []
        self.path = Path(path) if path else None
        self.print_every = print_every
        self._n = 0
        self._lock = threading.Lock()

    def update(self, cue: Cue, t: float = 0.0, frame: int = 0, hand_depth=None, target_depth=None,
               hand_xy=None, target_box=None):
        pan, freq, rate, mode = cue_to_params(cue, self.cfg)
        row = dict(t=round(t, 4), frame=frame, state=cue.state, dx=round(cue.dx, 4), dy=round(cue.dy, 4),
                   distance=round(cue.distance, 4), depth_offset=round(cue.depth_offset, 4),
                   depth_kind=cue.depth_kind,
                   hand_depth=None if hand_depth is None else round(hand_depth, 4),
                   target_depth=None if target_depth is None else round(target_depth, 4),
                   hand_x=None if hand_xy is None else round(hand_xy[0], 1),
                   hand_y=None if hand_xy is None else round(hand_xy[1], 1),
                   target_box=None if target_box is None else [round(v, 1) for v in target_box],
                   pan=round(pan, 3), freq_hz=round(freq, 1),
                   beep_rate=round(rate, 2), mode=mode)
        with self._lock:
            self.rows.append(row)
        self._n += 1
        if self.print_every and self._n % self.print_every == 0:
            print(f"[cue] f={frame} {cue.state:9s} dx={cue.dx:+.2f} dy={cue.dy:+.2f} dist={cue.distance:.2f} "
                  f"pan={pan:+.2f} {freq:.0f}Hz {rate:.1f}/s {mode}")

    def close(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                w.writeheader()
                w.writerows(self.rows)
