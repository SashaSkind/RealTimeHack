"""Sound scene classifier for the Sensory Agent: YAMNet (Google AudioSet, 521 classes, 4 MB TFLite).

Adapted from Quietcare's server/app/providers/audio_scene.py (same author). Two label sets:
- TRIGGERS: sensory-overload triggers from the hackathon brief (blender, sirens, alarms, dogs,
  babies crying, dishes, slams, beeps, fluorescent hum, overlapping voices ...). Exact YAMNet names.
- DISTRESS: Quietcare's substring list for a person in distress (scream, crying, groan, thud ...).

Scores are the MAX over YAMNet's ~0.48 s frames (Quietcare averaged them), so a short slam or
beep inside a longer window is not diluted.

  python -m sensory.audio_scene clip.wav        # classify a file
  python -m sensory.audio_scene --mic           # live: prints triggers every 0.5 s (mic permission)

Model files: models/yamnet/yamnet.tflite + yamnet_class_map.csv (copied from Quietcare; originals:
kagglehub "google/yamnet/tfLite/classification-tflite" and the tensorflow/models class map).
"""
from __future__ import annotations

import csv
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "yamnet"
SR = 16000

TRIGGERS = (
    "Blender", "Hair dryer", "Vacuum cleaner", "Microwave oven", "Mechanical fan",
    "Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
    "Civil defense siren", "Fire alarm", "Smoke detector, smoke alarm", "Alarm", "Alarm clock",
    "Car alarm", "Buzzer", "Bark", "Baby cry, infant cry", "Crying, sobbing",
    "Dishes, pots, and pans", "Clatter", "Slam", "Door", "Beep, bleep", "Reversing beeps",
    "Whistle", "Mains hum", "Hum", "Buzz", "Drill", "Power tool", "Jackhammer",
    "Chatter", "Crowd", "Hubbub, speech noise, speech babble", "Screaming", "Shout", "Yell",
    "Applause",
)

# Quietcare's distress keys, matched as case-insensitive substrings of YAMNet labels.
DISTRESS_KEYS = (
    "scream", "shout", "yell", "groan", "grunt", "crying", "sob", "wail",
    "moan", "gasp", "thud", "thump", "bang", "slam", "gunshot", "gunfire",
    "explosion", "firecracker", "fireworks", "glass", "shatter", "breaking",
    "fall", "whimper", "choking", "cough", "gagging",
)


@dataclass
class SceneResult:
    tags: list = field(default_factory=list)        # top labels overall [(label, score)]
    triggers: list = field(default_factory=list)    # trigger labels >= threshold
    distress: bool = False
    distress_tags: list = field(default_factory=list)
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    ms: float = 0.0

    def to_dict(self):
        r = lambda xs: [{"label": l, "score": round(s, 3)} for l, s in xs]
        return {"tags": r(self.tags), "triggers": r(self.triggers), "distress": self.distress,
                "distress_tags": r(self.distress_tags), "rms_dbfs": round(self.rms_dbfs, 1),
                "peak_dbfs": round(self.peak_dbfs, 1), "ms": round(self.ms, 1)}


def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite import Interpreter
    return Interpreter(model_path=str(path))


def load_labels(path) -> list[str]:
    with open(path, newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        i = hdr.index("display_name") if "display_name" in hdr else 2
        return [row[i] for row in rd if row]


def dbfs(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return -120.0, -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    peak = float(np.max(np.abs(x)))
    return 20 * np.log10(max(rms, 1e-6)), 20 * np.log10(max(peak, 1e-6))


def read_wav(path) -> np.ndarray:
    """Mono float32 at 16 kHz from a PCM WAV (linear resample, as in Quietcare)."""
    with wave.open(str(path), "rb") as wf:
        ch, sr, sw = wf.getnchannels(), wf.getframerate(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dt).astype(np.float32) / float(np.iinfo(dt).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR and x.size > 1:
        n = int(round(x.size * SR / sr))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return x


class AudioScene:
    def __init__(self, model_dir=MODEL_DIR, threshold: float = 0.30, distress_threshold: float = 0.30):
        model_dir = Path(model_dir)
        self.labels = load_labels(model_dir / "yamnet_class_map.csv")
        missing = [t for t in TRIGGERS if t not in self.labels]
        if missing:
            raise ValueError(f"trigger labels not in YAMNet class map: {missing}")
        self.trigger_idx = np.array([self.labels.index(t) for t in TRIGGERS])
        self.distress_idx = np.array([i for i, l in enumerate(self.labels)
                                      if any(k in l.lower() for k in DISTRESS_KEYS)])
        self.threshold, self.distress_threshold = threshold, distress_threshold
        self.it = _interpreter(model_dir / "yamnet.tflite")
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self._n = None

    def scores(self, x: np.ndarray) -> np.ndarray:
        """(frames, 521) class scores for a 16 kHz mono float32 waveform."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.size < 15600:  # YAMNet needs at least one 0.975 s window
            x = np.pad(x, (0, 15600 - x.size))
        if self._n != x.size:
            self.it.resize_tensor_input(self.inp["index"], [x.size])
            self.it.allocate_tensors()
            self._n = x.size
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])

    def classify(self, x: np.ndarray, top: int = 6) -> SceneResult:
        t = time.perf_counter()
        s = self.scores(x).max(axis=0)
        rms, peak = dbfs(x)
        order = np.argsort(s)[::-1]
        trig = [(self.labels[i], float(s[i])) for i in self.trigger_idx if s[i] >= self.threshold]
        dis = [(self.labels[i], float(s[i])) for i in self.distress_idx if s[i] >= self.distress_threshold]
        return SceneResult(
            tags=[(self.labels[i], float(s[i])) for i in order[:top]],
            triggers=sorted(trig, key=lambda p: -p[1]),
            distress=bool(dis), distress_tags=sorted(dis, key=lambda p: -p[1]),
            rms_dbfs=rms, peak_dbfs=peak, ms=(time.perf_counter() - t) * 1000)

    def classify_wav(self, path) -> SceneResult:
        return self.classify(read_wav(path))


def _live(scene: AudioScene, window_s=0.975, hop_s=0.5):
    """Mic -> rolling window -> print triggers. Opening the mic triggers the macOS permission prompt."""
    import queue

    import sounddevice as sd

    q, buf = queue.Queue(), np.zeros(0, np.float32)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(SR * hop_s),
                        callback=lambda d, *_: q.put(d[:, 0].copy())):
        print("listening... Ctrl-C to stop")
        while True:
            buf = np.concatenate([buf, q.get()])[-int(SR * window_s):]
            if buf.size < int(SR * window_s):
                continue
            r = scene.classify(buf, top=3)
            trig = ", ".join(f"{l} {s:.2f}" for l, s in r.triggers) or "-"
            flag = "  DISTRESS" if r.distress else ""
            print(f"{r.rms_dbfs:6.1f} dBFS  triggers: {trig}{flag}   (top: {r.tags[0][0]} {r.tags[0][1]:.2f}, {r.ms:.0f} ms)")


if __name__ == "__main__":
    import json
    import sys

    sc = AudioScene()
    if len(sys.argv) > 1 and sys.argv[1] == "--mic":
        try:
            _live(sc)
        except KeyboardInterrupt:
            pass
    else:
        for p in sys.argv[1:]:
            print(p, json.dumps(sc.classify_wav(p).to_dict(), indent=1))
