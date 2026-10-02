# Build timing log: 2026-09-29, 20:00–21:58

This is a wall-clock log, rebuilt from file timestamps because my running notes drifted. One agent (me) built the code. One helper subagent sourced and labelled the photos in parallel, from 20:02 to 20:29.

Machine: M3 Pro, 36 GB. It was on a **20 W** charger, which matters for the timings (see report §D).

| Clock | Minutes | Part | Notes for the 7-hour plan |
|---|---|---|---|
| 20:00–20:08 | 8 | venv + dependencies | Two traps. (1) `ultralytics` pulls in `opencv-python`, which hides the CSRT tracker; install `opencv-contrib-python` only. (2) `record3d` needs `brew install cmake`. |
| 20:02–20:24 | (bg) | Cosmos access check + GGUF downloads | The official repos are gated. The 2B download took 5 min and 8B Q8_0 took 15 min. **Download these before Friday.** |
| 20:05–20:08 | 3 | `timing.py`, `guidance.py`, Test A (23 tests) | Test A caught a real bug on its first run. |
| 20:08–20:25 | 17 | sources, hand, target, depth, audio, speech, slow_loop, main | This is all the code. Most of the time went to the audio synth and the slow-loop threading. |
| 20:21–20:24 | 3 | `llama-server` + OpenAI-compatible client smoke test | llama.cpp from Homebrew (0.5.0) serves Qwen3-VL/Cosmos with images out of the box. |
| 20:22–20:27 | 5 | Searched Commons for an openly licensed reaching clip | Found 2 CC BY clips. |
| 20:27–20:30 | 3 | First end-to-end run | mediapipe 1.0.1 crashes on macOS, so pin `mediapipe<1`. Fixed the pacing clock starting before warm-up. |
| 20:02–20:29 | (27, parallel) | Test B data: 48 photos + labels (subagent) | Delegating this was the biggest time saver. I spent 5 min spot-checking every box on contact sheets. |
| 20:30–20:35 | 5 | Tests for sources, audio, speech, slow loop | Fixed 2 racy tests. |
| 20:34–20:46 | 12 | Test B: 5 models × 3 prompt sets + prompt-word sweep | About 4 min of compute per full sweep. |
| 20:46–20:48 | 2 | Detector gating against Cosmos boxes + tracker tests | |
| 20:46–21:07 | 21 | Test C: eval script + 6 runs (8B/2B, prompt/schema, pixel/norm) | 8B took about 3.5 min per 30 images. |
| 20:49–20:53 | 4 | Test E script + the rotation trick for the y axis | |
| 20:49–20:59 | 10 | LIVE_TESTS.md, RECORDING_CHECKLIST.md, README | |
| 21:07–21:20 | 13 | Test C iterations: runaway reply fix, box-first prompt, affordance hints | |
| 21:15–21:41 | 26 | **Lost to diagnosing slow timings** | First the aerial screensaver taking the GPU, then memory pressure from two model servers, then the real cause: a 20 W charger with the battery at 8% and the GPU throttled. |
| 21:42–21:52 | 10 | Test D latency runs + re-runs; `--search-every`; speech gate; hold-on-occlusion | |
| 21:52–21:58 | 6 | Re-ran detector accuracy and Test E after the tracker changes; licence audit | |
| 21:58– | | This log + report | |

**Total: about 2 h of wall clock for everything here.** Without the power problem it would have been about 1.5 h.

## Suggested split for the 7-hour hackathon

The code exists now, so Friday is integration and live tuning:

| Time | Task |
|---|---|
| 0:00–0:30 | Point `--cosmos-url` at CoreWeave. Run `bench/cosmos_eval.py` against it: it's the same script, and it gives real remote latency in 5 minutes. |
| 0:30–1:30 | Record3D live (LIVE_TESTS §0 and test 2). Record the clips on RECORDING_CHECKLIST. |
| 1:30–3:30 | Live reach tuning: dead zone, beep curve, hold/touch thresholds, camera mount. This is the part nobody has tested. |
| 3:30–5:00 | Door flow end-to-end with Cosmos (test 3): speech timing, box staleness. |
| 5:00–6:00 | Re-run Tests D and E on your own clips. Update the slides. |
| 6:00–7:00 | Buffer and demo rehearsal. |
