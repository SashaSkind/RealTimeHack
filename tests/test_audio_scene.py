"""YAMNet sound scene: triggers from the brief exist in the model, silence is quiet, beeps trigger."""
import numpy as np
import pytest

from sensory.audio_scene import MODEL_DIR, SR, TRIGGERS, dbfs

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "yamnet.tflite").exists(),
                                reason="models/yamnet not present (copy from Quietcare or fetch, see sensory/audio_scene.py)")


@pytest.fixture(scope="module")
def scene():
    from sensory.audio_scene import AudioScene
    return AudioScene()


def test_every_brief_trigger_is_a_yamnet_class(scene):
    assert len(scene.trigger_idx) == len(TRIGGERS)
    for t in ("Blender", "Fire alarm", "Bark", "Baby cry, infant cry", "Vacuum cleaner", "Mains hum"):
        assert t in TRIGGERS


def test_silence_has_no_triggers(scene):
    r = scene.classify(np.zeros(SR * 2, np.float32))
    assert r.triggers == [] and not r.distress and r.rms_dbfs <= -100


def test_beep_pattern_is_a_beep_trigger(scene):
    t = np.arange(SR * 2) / SR
    x = (0.5 * np.sin(2 * np.pi * 2500 * t) * ((t * 4) % 1 < 0.5)).astype(np.float32)
    r = scene.classify(x)
    assert ("Beep, bleep" in [l for l, _ in r.triggers]), r.tags


def test_short_clip_is_padded(scene):
    r = scene.classify(np.zeros(4000, np.float32))
    assert r.tags


def test_dbfs():
    assert dbfs(np.ones(100, np.float32))[0] == pytest.approx(0.0, abs=1e-6)
    assert dbfs(np.full(100, 0.1, np.float32))[0] == pytest.approx(-20.0, abs=1e-3)
