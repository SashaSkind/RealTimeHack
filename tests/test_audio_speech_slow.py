"""Unit tests for audio synthesis, the speech wrapper, and the slow loop (with a fake server)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from handguide.audio import AudioConfig, ToneSynth, cue_to_params
from handguide.guidance import Cue
from handguide.slow_loop import CosmosClient, LatestFrame, SlowLoop, box_to_pixels, extract_json, validate
from handguide.sources import Frame
from handguide.speech import Speaker


# --- audio ---------------------------------------------------------------------

def _max_step(x):
    return float(np.max(np.abs(np.diff(x, axis=0))))


def test_pan_follows_dx():
    s = ToneSynth()
    s.set_cue(Cue(dx=1.0, dy=0.0, distance=0.5, state="guiding"))
    out = np.concatenate([s.render(512) for _ in range(40)])
    tail = out[-4800:]
    assert np.abs(tail[:, 1]).sum() > 5 * np.abs(tail[:, 0]).sum(), "dx=+1 must be louder in the RIGHT channel"
    s.set_cue(Cue(dx=-1.0, dy=0.0, distance=0.5, state="guiding"))
    out = np.concatenate([s.render(512) for _ in range(40)])
    tail = out[-4800:]
    assert np.abs(tail[:, 0]).sum() > 5 * np.abs(tail[:, 1]).sum()


def test_pitch_up_for_positive_dy():
    cfg = AudioConfig()
    up = cue_to_params(Cue(dy=1.0, state="guiding"), cfg)[1]
    down = cue_to_params(Cue(dy=-1.0, state="guiding"), cfg)[1]
    assert up > cfg.base_hz > down


def test_beep_rate_faster_when_closer():
    cfg = AudioConfig()
    far = cue_to_params(Cue(distance=1.0, state="guiding"), cfg)[2]
    near = cue_to_params(Cue(distance=0.05, state="close"), cfg)[2]
    assert near > far


def test_touching_is_steady_and_searching_is_silent():
    cfg = AudioConfig()
    assert cue_to_params(Cue(state="touching"), cfg)[3] == "steady"
    assert cue_to_params(Cue(state="searching"), cfg)[3] == "silent"


def test_no_clicks_on_abrupt_parameter_jumps():
    """Worst case: jump pan -1 -> +1, pitch -1 -> +1 octave, rate, and state every 50 ms.
    A click shows up as a sample-to-sample step far above what a smooth sine can produce."""
    cfg = AudioConfig()
    s = ToneSynth(cfg)
    rng = np.random.default_rng(0)
    blocks = []
    states = ["guiding", "close", "touching", "lost", "searching"]
    for i in range(200):
        if i % 10 == 0:
            s.set_cue(Cue(dx=rng.choice([-1, 1]), dy=rng.choice([-1, 1]), distance=rng.random(),
                          state=states[(i // 10) % len(states)]))
        blocks.append(s.render(256))
    out = np.concatenate(blocks)
    fmax = cfg.base_hz * 2 ** cfg.octaves
    sine_step = 2 * np.pi * fmax / cfg.sr * cfg.volume      # max step of a full-volume sine at fmax
    env_step = cfg.volume / (cfg.edge_s * cfg.sr)           # max step from the beep edge
    assert _max_step(out) < 1.1 * (sine_step + env_step), _max_step(out)


# --- speech ----------------------------------------------------------------------

def test_speaker_never_blocks_and_keeps_only_newest():
    sp = Speaker(interrupt=False, log=None)
    sp._cmd = lambda text: ["sh", "-c", "sleep 0.3"]  # stand-in for `say`, no sound
    sp.speak("msg 0")
    time.sleep(0.1)  # worker has started "msg 0"
    t = time.perf_counter()
    for i in range(1, 5):
        sp.speak(f"msg {i}")
    assert time.perf_counter() - t < 0.05, "speak() must not block"
    time.sleep(1.0)
    spoken = [x for _, x in sp.spoken]
    assert spoken[0] == "msg 0"
    assert spoken[-1] == "msg 4"
    assert spoken == ["msg 0", "msg 4"], f"msgs 1-3 were stale and must be dropped; got {spoken}"
    sp.close()


def test_speaker_interrupts_stale_message():
    sp = Speaker(interrupt=True, log=None)
    sp._cmd = lambda text: ["sh", "-c", "sleep 2"]
    sp.speak("old")
    time.sleep(0.2)
    t = time.perf_counter()
    sp.speak("new")
    while [x for _, x in sp.spoken][-1] != "new" and time.perf_counter() - t < 1.5:
        time.sleep(0.02)
    assert [x for _, x in sp.spoken][-1] == "new"
    assert time.perf_counter() - t < 1.0, "new message should start without waiting for the old one"
    sp.close()


# --- slow loop parsing ------------------------------------------------------------

def test_extract_json_handles_fences_and_think():
    raw = '<think>let me see</think>\n```json\n{"target":"cup","affordance":"mug, grab handle","instruction":"Cup ahead","box":[1,2,3,4],"phase":"approaching"}\n```'
    d = validate(extract_json(raw))
    assert d["target"] == "cup" and d["box"] == [1.0, 2.0, 3.0, 4.0]


def test_validate_rejects_bad_box_and_fixes_phase():
    with pytest.raises(ValueError):
        validate({"target": "a", "affordance": "b", "instruction": "c", "box": [1, 2], "phase": "done"})
    d = validate({"target": "a", "affordance": "b", "instruction": "c", "box": None, "phase": "weird"})
    assert d["phase"] == "searching" and d["box"] is None


def test_box_to_pixels_norm1000_and_pixel():
    assert box_to_pixels([100, 200, 300, 400], "norm1000", (640, 480), (1280, 960)) == (128.0, 192.0, 384.0, 384.0)
    assert box_to_pixels([64, 48, 128, 96], "pixel", (640, 480), (1280, 960)) == (128.0, 96.0, 256.0, 192.0)
    assert box_to_pixels([300, 400, 100, 200], "norm1000", (640, 480), (1000, 1000)) == (100.0, 200.0, 300.0, 400.0)
    assert box_to_pixels([5, 5, 5, 5], "norm1000", (640, 480), (1000, 1000)) is None


class _FakeHandler(BaseHTTPRequestHandler):
    mode = "ok"
    delay = 0.0
    calls = 0

    def do_POST(self):
        _FakeHandler.calls += 1
        n = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(n))
        time.sleep(_FakeHandler.delay)
        if _FakeHandler.mode == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        content = "not json at all" if _FakeHandler.mode == "garbage" else json.dumps(
            {"target": "cup", "affordance": "mug, grab the handle", "instruction": f"Cup is ahead {_FakeHandler.calls // 3}",
             "box": [100, 100, 200, 200], "phase": "approaching"})
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    _FakeHandler.mode, _FakeHandler.delay, _FakeHandler.calls = "ok", 0.0, 0
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _frames():
    lf = LatestFrame()
    lf.put(Frame(ts=1.0, rgb=np.zeros((480, 640, 3), np.uint8)))
    return lf


def test_slow_loop_delivers_results_without_blocking(fake_server):
    got = []
    lf = _frames()
    loop = SlowLoop(CosmosClient(fake_server, json_mode="prompt"), lf, "pick up the cup", interval_s=0.05,
                    on_result=got.append, log=lambda *a: None)
    loop.start()
    for i in range(20):  # the "fast loop" keeps publishing frames and never waits
        t = time.perf_counter()
        lf.put(Frame(ts=2.0 + i, rgb=np.zeros((480, 640, 3), np.uint8)))
        assert loop.latest() is None or loop.latest().ok
        assert time.perf_counter() - t < 0.01
        time.sleep(0.03)
    loop.stop()
    assert got and got[-1].target == "cup"
    assert got[-1].box == (64.0, 48.0, 128.0, 96.0)  # norm1000 -> 640x480


def test_slow_loop_survives_errors_and_keeps_last_good(fake_server):
    lf = _frames()
    loop = SlowLoop(CosmosClient(fake_server, json_mode="prompt", timeout_s=0.3), lf, "g", interval_s=0.02,
                    log=lambda *a: None)
    loop.start()
    time.sleep(0.3)
    good = loop.latest()
    assert good is not None and good.ok
    for mode in ("500", "garbage"):
        _FakeHandler.mode = mode
        for i in range(5):
            lf.put(Frame(ts=100.0 + i + (10 if mode == "garbage" else 0), rgb=np.zeros((480, 640, 3), np.uint8)))
            time.sleep(0.05)
    _FakeHandler.mode, _FakeHandler.delay = "ok", 1.0  # slower than the client timeout
    lf.put(Frame(ts=500.0, rgb=np.zeros((480, 640, 3), np.uint8)))
    time.sleep(0.6)
    loop.stop()
    assert loop.failures >= 3
    assert loop.latest() is good or loop.latest().ok


def test_salvage_truncated_runaway_reply():
    from handguide.slow_loop import salvage
    raw = ('```json\n{\n  "target": "cup",\n  "affordance": "white ceramic cup with handle",\n  "instruction": '
           '"Guide your hand to the white cup on the left. Then grasp it firmly by the handle to lift it off the surf')
    d = validate(salvage(raw))
    assert d["target"] == "cup" and d["instruction"] == "Guide your hand to the white cup on the left."
    assert d["box"] is None


def test_speech_gate_ignores_rewording_but_speaks_real_changes():
    from handguide.speech import SpeechGate
    g = SpeechGate(min_change=0.6, min_gap_s=4.0, repeat_s=15.0)
    assert g.should_speak("grab the toy house on the stand", "approaching", now=0)
    assert not g.should_speak("grab the toy house on the stand.", "approaching", now=2)   # same words
    assert not g.should_speak("grasp the toy house on the stand", "approaching", now=5)   # reworded
    assert g.should_speak("press the lever down", "approaching", now=6)                  # new content
    assert not g.should_speak("turn the knob", "approaching", now=7)                     # too soon
    assert g.should_speak("press the lever down now", "touching", now=8)                 # phase change
    assert g.should_speak("press the lever down now", "touching", now=30)                # periodic reminder
