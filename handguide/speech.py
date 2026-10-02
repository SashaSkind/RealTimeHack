"""Non-blocking macOS `say` wrapper with a queue of at most one message.

- speak(text) never blocks.
- If a message is already waiting, the new one replaces it (the stale one is dropped).
- If `interrupt=True` (default), a new message also stops the one currently being spoken,
  because stale guidance is worse than a cut-off sentence.
"""
from __future__ import annotations

import difflib
import subprocess
import threading
import time


class SpeechGate:
    """Decides whether a new Cosmos instruction is worth saying. Cosmos rewords its answer on
    almost every call ("grab the toy house" -> "grasp the toy car", Test D slow-loop run), so plain
    string inequality would talk every 1.5 s. Speak when the phase changes, when the wording is
    substantially different (similarity < `min_change`), or when `repeat_s` has passed."""

    def __init__(self, min_change: float = 0.6, min_gap_s: float = 4.0, repeat_s: float = 15.0):
        self.min_change, self.min_gap_s, self.repeat_s = min_change, min_gap_s, repeat_s
        self.last_text, self.last_phase, self.last_t = None, None, -1e9

    def should_speak(self, text: str, phase: str = "", now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not text:
            return False
        if self.last_text is None:
            ok = True
        else:
            sim = difflib.SequenceMatcher(None, self.last_text.lower(), text.lower()).ratio()
            gap = now - self.last_t
            ok = (phase != self.last_phase and gap >= 1.0) or (sim < self.min_change and gap >= self.min_gap_s) \
                or gap >= self.repeat_s
        if ok:
            self.last_text, self.last_phase, self.last_t = text, phase, now
        return ok


class Speaker:
    def __init__(self, voice: str | None = None, rate: int | None = 200, interrupt: bool = True,
                 enabled: bool = True, cmd_prefix: list[str] | None = None, log=print):
        self.voice, self.rate, self.interrupt, self.enabled = voice, rate, interrupt, enabled
        self.cmd_prefix = cmd_prefix  # for tests, e.g. ["sleep", "0.5", ";"] style fakes
        self.log = log
        self._pending: str | None = None
        self._proc: subprocess.Popen | None = None
        self._cv = threading.Condition()
        self._stop = False
        self.spoken: list[tuple[float, str]] = []   # (time, text) actually started
        self.dropped: list[str] = []
        self._th = threading.Thread(target=self._run, daemon=True, name="speaker")
        self._th.start()

    def _cmd(self, text: str) -> list[str]:
        if self.cmd_prefix is not None:
            return [*self.cmd_prefix, text]
        cmd = ["say"]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.rate:
            cmd += ["-r", str(self.rate)]
        return cmd + [text]

    def speak(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        with self._cv:
            if self._pending is not None:
                self.dropped.append(self._pending)
            self._pending = text
            if self.interrupt and self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                text, self._pending = self._pending, None
            self.spoken.append((time.time(), text))
            if not self.enabled:
                if self.log:
                    self.log(f"[say] {text}")
                continue
            try:
                proc = subprocess.Popen(self._cmd(text), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                if self.log:
                    self.log(f"[say unavailable] {text}")
                continue
            with self._cv:
                self._proc = proc
            proc.wait()

    def busy(self) -> bool:
        return self._pending is not None or (self._proc is not None and self._proc.poll() is None)

    def close(self):
        with self._cv:
            self._stop = True
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._cv.notify()
