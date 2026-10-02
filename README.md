# handguide

This is audio guidance of a hand to an object, for blind and low-vision users.

- **Fast loop** (every frame): MediaPipe palm centre, then YOLOE with a CSRT tracker for the target, then depth (LiDAR, or Depth Anything V2 Small), then guidance, then a stereo tone.
  - Pan tells you left or right.
  - Pitch tells you up or down.
  - Beep rate tells you distance.
  - A steady tone means you are touching.
- **Slow loop** (every 1–2 s, in its own thread): the newest frame goes to Cosmos Reason 2 over any OpenAI-compatible endpoint. The reply is strict JSON: `{target, box, affordance, instruction, phase}`.
  - The box resets the tracker.
  - A changed instruction is spoken with macOS `say`.
  - The fast loop never waits for it.

## Setup
```bash
brew install cmake llama.cpp
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
python -m pytest -q
```
Model files go in `models/` (not in git). The MediaPipe hand model and the local Cosmos GGUFs: see `bench/fetch_models.py`. YOLO weights and Depth Anything download on first use. Free iPhone LiDAR streaming (`--source pathon`): see the optional section of `requirements.txt`.

## Run
```bash
# offline replay (paced like a live camera), cues logged instead of played
python -m handguide.main --source video --path data/clips/reach_bottle.ogv --target bottle --no-audio \
    --save-video output/runs/demo/overlay.mp4 --log-dir output/runs/demo
# live
python -m handguide.main --source webcam --target cup --display
python -m handguide.main --source record3d --target "door handle" --goal "open the door" --cosmos-url $COSMOS_URL --display
```
See `python -m handguide.main -h` for all options. For tomorrow's hands-on checks see `LIVE_TESTS.md`; for clips to record see `RECORDING_CHECKLIST.md`. Results are in `output/report.md`.

## Layout
```
handguide/  sources, hand, target, depth, guidance, audio, slow_loop, speech, timing, main
tests/      unit tests (guidance math = Test A)
bench/      detector_bench, prompt_search, cosmos_eval, latency_runs.sh, open_loop_check, make_synthetic_video, audio_demo
data/       photos/ (48 labelled Commons photos + labels.csv with licences), clips/ (CC BY reach clips + sources)
output/     report.md, build_timing_log.md, bench/, runs/, audio_demo.wav
```
