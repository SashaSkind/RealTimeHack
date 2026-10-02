"""Renders output/audio_demo.wav: what a reach sounds like, from the real ToneSynth.
0-1 s searching (silence), 1-2 s hand lost (slow low pulse), 2-8 s hand moves from the
lower-left toward a target up and to the right, 8-9 s close, 9-11 s touching (steady tone).
Listen with headphones: the target is up and to the right, so the tone starts hard right and
high, then slides to centre and base pitch while the beeps speed up; see the printed timeline."""
from pathlib import Path
import numpy as np
from handguide.audio import AudioConfig, ToneSynth, write_wav
from handguide.guidance import Guide

cfg = AudioConfig()
s = ToneSynth(cfg)
g = Guide()
W, H, fps = 640, 480, 30
target = (480, 100, 520, 140)
blocks = []
for f in range(11 * fps):
    t = f / fps
    if t < 1:
        cue = g.update((100, 400), None, (W, H))
    elif t < 2:
        cue = g.update(None, target, (W, H))
    else:
        a = min(1.0, (t - 2) / 7.0)
        hx, hy = 100 + a * (500 - 100), 400 + a * (120 - 400)
        cue = g.update((hx, hy), target, (W, H))
    s.set_cue(cue)
    if f % 15 == 0:
        print(f"{t:4.1f}s {cue.state:9s} dx={cue.dx:+.2f} dy={cue.dy:+.2f} dist={cue.distance:.2f}")
    blocks.append(s.render(cfg.sr // fps))
out = Path("output/audio_demo.wav")
write_wav(out, np.concatenate(blocks), cfg.sr)
print("wrote", out)
