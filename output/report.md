# HandGuide: overnight build and test report (2026-09-29)

## Summary

| Test | Result | Why |
|---|---|---|
| **A. Guidance maths** | **PASS** | 23/23 guidance tests pass. All 55 unit tests in the repo pass. The up/down and left/right cases are checked in image coordinates. The first run caught a real bug (distance saturated at half a frame), which is now fixed. |
| **B. Detector on real photos** | **FAIL for door hardware, PASS for cups** | On 48 labelled Commons photos, cups were found 10/10 by every model. Door hardware and switches were not reliable with any model: best loose top-1 hit rates were levers 50–60%, knobs 30–60%, cabinet handles 0–33%, switches 33–89%, with false-fire rates of 20–50%. **Pick: YOLOE-11s + tuned prompts.** On doors, the Cosmos box has to be the primary target source. |
| **C. Cosmos locally** | **PASS, with a substitution** | The official NVIDIA repos are gated for your HF account, so I ran third-party Q8_0 GGUFs under llama.cpp. 8B, final prompt: JSON 30/30, target 30/30, correct operation 29/30, box IoU ≥ 0.5 25/30, **5.6–6.9 s per call on this Mac**. 2B: 1.25 s, but a clearly worse operation answer. MLX and `transformers` were not tried, because both need the gated weights. |
| **D. Per-stage latency** | **RUN, numbers pessimistic** | Hand + target present, detect every 5 frames + CSRT + Depth Anything: **59 ms median / 109 ms p95 per frame; 89 / 142 ms frame→cue**. Detect every frame: 69 / 86 ms. The Mac was on a **20 W charger at 8% battery**, and GPU work (YOLOE) ran ~2× slower than earlier in the night. |
| **E. Open-loop sanity** | **PASS, weak evidence** | Cue direction agreed with the actual hand motion on 86/86 scored frames. That covers 2 CC BY clips, with 51 of those frames on the y axis using rotated copies; a sign-flipped control scores 0%. It is only 2 short clips, and it **does not show a person can follow the cues**. |
| Live 1–3 (webcam reach, iPhone depth, door + Cosmos) | **NOT RUN** | These need you, a camera and the phone. See `LIVE_TESTS.md`. |

**Things that went wrong tonight, most important first:**
1. **This Mac is on a 20 W USB-C charger.** Sustained GPU load drained the battery to 8% and throttled the GPU. I lost ~25 minutes finding this, and every timing after ~21:10 had to be discarded or caveated. **Bring the real ≥70 W charger on Friday.**
2. **The Cosmos models are gated** for account `SashaSk`. See "What you need to click" below.
3. **`mediapipe` 1.0.1 crashes on macOS** (Metal "Service is unavailable"). It is pinned to 0.10.35 with the CPU delegate.
4. **`ultralytics` installs `opencv-python`,** which hides the CSRT tracker. Install `opencv-contrib-python` only.

### What you need to click
**Done (22:05):** licence accepted for both models; access verified from this Mac.

Go to <https://huggingface.co/nvidia/Cosmos-Reason2-8B> and <https://huggingface.co/nvidia/Cosmos-Reason2-2B>, logged in as **SashaSk**, and press **"Agree and access repository"** to accept the NVIDIA Open Model License. You only need this for the official safetensors: an MLX conversion or `transformers` on MPS. You don't need it for Friday if CoreWeave serves the model.

The GGUFs I used are redistributions under the same NVIDIA Open Model License. By using them you are bound by that licence exactly as with the originals.

---

## Model and prompt choices for Friday

| Piece | Choice | Why |
|---|---|---|
| Hand | MediaPipe hand landmarker, palm centre (wrist + 4 MCPs), CPU delegate | 20 ms per frame; stable point |
| Target detector | **YOLOE-11s** (`yoloe-11s-seg.pt`), imgsz 640, conf 0.1, detect every 5 frames, **CSRT** in between (KCF if fps is tight) | Best accuracy/speed of the 5 tried (27 ms on MPS, clean) |
| Prompts | cup → `cup, mug`; door handle → `handle, lever handle`; knob → `round knob, door knob`; switch → `switch, switch plate`; cabinet → `cabinet handle, drawer handle` | Literal names often lose to generic words: `door handle` hit 10% vs `handle` 50%; `light switch` 0% vs `switch` 56%. |
| Door targets | The **Cosmos box resets the tracker**, and the detector may only refine a box that overlaps it | The detector is too weak on door hardware to be trusted alone |
| Cosmos | **8B**, `--json-mode schema`, box in **0–1000 normalised** coordinates, **box before the text fields**, **affordance hint on** | See Test C. The 2B model needs `--no-cosmos-hints`. |
| Slow-loop interval | 1.5 s (`--slow-interval`). Calls never overlap; if a call takes longer, the next starts as soon as it returns. | |
| Depth | LiDAR via Record3D when present; otherwise Depth Anything V2 Small at 252 px, or `--depth off` | |

---

## Test A: guidance maths (PASS)

`python -m pytest tests/test_guidance.py -q` gives **23 passed**. Conventions: image y grows down; cue `dy > 0` means **move up**; `dx > 0` means move right.

The cases covered:
- **Direction:** target right → right; target left → left; target **above → up (`dy > 0`)**; target **below → down**; diagonal up-left; clipping to [−1, 1].
- **Distance:** near < far; near gives `close`; distance decreases monotonically as the hand approaches.
- **Dead zone:** small offsets go to exactly 0; the zone is continuous (no jump at its edge).
- **On target:** gives `touching` with no depth.
- **Depth:**
  - On target in 2D but 30 cm in front → **not** `touching`, and the cue says reach forward.
  - Matching LiDAR depth → `touching`.
  - Pull-back sign is right.
  - Relative-depth sign is right (Depth Anything is inverse depth).
  - Missing or zero depth is ignored.
- **Missing inputs:** hand missing → `lost`; target missing → `searching`; both missing → `searching`.
- **Stateful wrapper:** holds the last cue for a few frames before `lost`; `touching` needs 3 consecutive frames; smoothing keeps the sign.

The bug Test A found: distance used the clipped half-frame offsets, so it sat at 1.0 for any hand more than half a frame away. The beep rate would have given no feedback across half the screen. Distance is now normalised by full frame size.

Other unit tests (32 more, all passing) cover:
- **Audio:** pan follows `dx`; pitch rises for up; beeps get faster when closer; no clicks under worst-case parameter jumps (max sample step within a smooth sine's bound).
- **Speech:** never blocks; drops stale messages; interrupts the current message.
- **Slow loop:** JSON from fences or `<think>`; truncated-reply salvage; box conversion; against a fake HTTP server it survives HTTP 500, garbage replies and timeouts, keeps the last good state, and never blocks the caller.
- **Sources:** paced video really drops frames when processing is slow; record → replay round-trip with depth; Record3D with no phone raises a clean error and `main` exits with code 2.
- **Tracker:** detect-every-N; external box with stale-frame compensation; gating; hold during occlusion.

## Test B: detector on real photos (FAIL for doors, PASS for cups)

**Data:** 48 photos from Wikimedia Commons, all CC0, public domain, CC BY or CC BY-SA. They cover 10 lever handles, 10 knobs, 9 cabinet handles, 9 light switches and 10 cups, with 21 close-ups and 27 cluttered or small scenes. URL, licence and author are in `data/photos/labels.csv`. A subagent drew the boxes on a 64-px grid overlay and checked each one on a render; I then checked all 48 on contact sheets. They are rough boxes. Light-switch boxes cover the whole plate.

**Method:**
- Only the target's prompts are active, which is how the app runs.
- **hit** means the top detection has IoU ≥ 0.5.
- **loose** means top detection IoU ≥ 0.3, or its centre lies inside the label.
- **fp/img** counts unmatched detections on positive images. cabinet_handle_08 and _09 are excluded because they have many unlabelled identical handles.
- **neg_fire** is the share of unrelated-category images where the prompt fires at all.

Speeds were measured at 20:35, before the power problem. Full tables are in `output/bench/detector_results.md`.

**Summary at conf 0.1, tuned prompts, mean over 5 categories:**

| Model | hit | loose | fp/img | neg_fire | MPS ms (med/p95) | CPU ms |
|---|---|---|---|---|---|---|
| YOLO-World v2-S | 0.26 | 0.28 | 0.36 | 0.07 | 25 / 31 | 63 |
| YOLO-World v2-M | 0.24 | 0.26 | 0.24 | 0.02 | 26 / 31 | 136 |
| **YOLOE-11S** | **0.32** | **0.43** | 0.46 | 0.22 | **27 / 35** | 84 |
| YOLOE-11M | 0.35 | 0.58 | 0.45 | 0.26 | 36 / 42 | 188 |
| YOLOE-26S | 0.30 | 0.32 | 0.26 | 0.12 | 29 / 35 | 87 |

**Per category, YOLOE-11S tuned (conf 0.1 / conf 0.05), loose top-1 hit:**

| Category | Loose @0.1 | Loose @0.05 | neg_fire @0.1 |
|---|---|---|---|
| cup | **1.00** | 1.00 | 0.13 |
| lever handle | 0.50 | 0.60 | 0.18 |
| door knob | 0.30 | 0.60 | 0.36 |
| light switch | 0.33 | 0.56 | 0.10 |
| cabinet handle | 0.00 | 0.33 | 0.31 |

**Why YOLOE-11S:**
- It has the best loose hit rate of the fast models: 0.43 vs 0.28 for YOLO-World-S, at the same ~27 ms.
- YOLOE-11M is a bit better (0.58, and it reaches 0.89 on switches) but costs +9 ms per detection on MPS and more than 2× on CPU. Since detection runs every 5th frame, **YOLOE-11M is a reasonable swap** (`--detector yoloe-m`) if light switches matter for the demo.
- YOLO-World fires less on negatives but misses almost all door hardware.

**Prompt wording mattered more than model choice** (`bench/prompt_search.py`, YOLOE-11S, 640, conf 0.05, loose hit / neg_fire):

| Category | Prompt | Loose hit | neg_fire |
|---|---|---|---|
| Lever | `door handle` | 0.10 | 0.21 |
| Lever | `handle` | 0.50 | 0.04 |
| Lever | `lever handle` | 0.60 | 0.43 |
| Switch | `light switch` | 0.00 | — |
| Switch | `switch` | 0.56 | — |
| Knob | `door knob` | 0.00 | — |
| Knob | `doorknob` | 0.00 | — |
| Knob | `round knob` | 0.60 | 0.32 |

**Higher input resolution** (1024 vs 640) helped levers and switches a little on YOLOE-11S, but raised false fires. I kept 640.

**What this means:**
- For **cups**, the fast loop can find the target by itself.
- For **doors**, it can't reliably. So the Cosmos box seeds the tracker, and the detector may only refine a box that overlaps the Cosmos box. It cannot move the target elsewhere (`TargetTracker._pick`).
- Caveat: these are single still photos. Live, the detector gets many attempts per second, and Cosmos supplies a prior.

## Test C: Cosmos Reason 2 locally (PASS with substitution)

**Setup:**

| Item | Detail |
|---|---|
| Weights | `nvidia/Cosmos-Reason2-8B` and `-2B` are **gated** (403 for SashaSk). Following your order of attempts: **MLX** has no Reason2 conversion on the Hub, and making one needs the gated weights, so not run. **GGUF works**, via third-party redistributions under the NVIDIA Open Model License: `mradermacher/Cosmos-Reason2-8B-GGUF` (Q8_0 + mmproj-f16) and `apolo13x/Cosmos-Reason2-2B-GGUF` (Q8_0 + mmproj-F16). `transformers` on MPS needs the gated weights, so not run. |
| Server | `llama-server` (Homebrew llama.cpp 0.5.0), which exposes **an OpenAI-compatible `/v1/chat/completions`**. `slow_loop.py` talks to it exactly as it will to vLLM on Friday. |
| Substitution | Q8_0 quantisation, not bf16. Expect small quality differences on Friday. |

**Prompt:**
- Strict JSON, no reasoning.
- Goal "open the door" or "pick up the cup".
- The image is downscaled to 640 px on its long side.
- Temperature 0 and `max_tokens` 200.
- Scored on the 30 lever, knob and cup photos.
- "Operation correct" is **my manual judgement** of the affordance and instruction: lever → press/pull down; knob → turn; cup → grip/lift. The keyword counts are in the JSON files.

| Run (all on 30 images) | JSON parsed | Target right | Operation right | Box IoU ≥ 0.5 | Median IoU | s/call (median / p95) |
|---|---|---|---|---|---|---|
| 8B, first prompt, prompt-only JSON | 29 | 29 | 23 | 24 | 0.89 | 7.2 / 7.7 |
| 8B, box-first + schema | 30 | 30 | 22 (levers 3/10) | 26 | 0.91 | **5.6 / 5.8** |
| **8B, box-first + schema + affordance hint (final)** | **30** | **30** | **29** (levers 9/10) | 25 | 0.89 | ~6.9 (see note) |
| 2B, box-first + schema | 30 | 28 | 18 (levers 0/10: "grab the handle") | 24 | 0.78 | **1.25 / 1.40** |
| 2B + affordance hint | 30 | 22 | 22, but copies the hint: 8/10 knobs called "lever door handle" | 21 | 0.73 | throttled |
| 8B, box requested in *pixels* | 30 | 30 | — | **0** (27 if read as 0–1000) | — | 5.6 |
| 8B box-first + schema, **no hint**, cabinet handles + switches (18 images) | 18 | 14 (4 drawers named "drawer", not the handle) | 14 (cabinet 5/9: 4 say "pull down"; switches 9/9) | 11 | 0.89 | 6.7 / 7.1 |

The hinted run on cabinet handles and switches was started but killed during the power problem, so the hint is **not** tested on those categories.

Timing note: the 8B-with-hint run finished at 21:10, just as throttling started. Its answers are identical, token for token, to a later re-run; its time may be slightly high.

**Findings:**
- **The box is in 0–1000 normalised coordinates**, which is Qwen3-VL's native format, even when you ask for pixels. `box_to_pixels(..., "norm1000")` is the default.
- **The 8B box is usable:** median IoU 0.89–0.91 against my rough labels. The misses are small, far-away handles (lever_handle_01 and _10 at IoU ~0.13–0.2) and one knob.
- **Without a hint, 8B says "turn clockwise" for most lever handles.** With one hint sentence in the system prompt, it goes to 9/10 while still telling knobs ("grip and turn") apart from levers. A shorter hint did *not* work. 2B copies the hint blindly, so use `--no-cosmos-hints` with 2B.
- **A runaway reply happened once:** a 700-character instruction hit `max_tokens` and broke the JSON. Fixed three ways:
  - `maxLength` on the schema strings; llama.cpp and vLLM grammars both honour it.
  - The box is now emitted *before* the text fields, so it survives truncation.
  - A regex salvage parser as a last resort.
- **Latency breakdown for 8B on this Mac:** prefill ~1.7–1.9 s for ~400–450 tokens (~250 tok/s), decode ~15 tok/s. The ~60–80 output tokens dominate at ~4–5 s. **On Friday's H100 with vLLM, I estimate ~0.7–1.2 s per call. This is not measured.** The first thing to do on Friday is `python bench/cosmos_eval.py --url $COSMOS_URL --model $COSMOS_MODEL --tag friday_8b --json-mode schema` (5 min).
- **2B vs 8B on one image:** the full 30-image comparison is above. For example, cup_05: 8B gives IoU 0.86, "grasp the cup's handle to lift it from the saucer", in 5.6 s. 2B gives IoU 0.63, "grip the handle of the coffee cup", in 1.3 s.

Raw replies for every run are in `output/bench/cosmos/*.json`. Runs whose timings were hit by the power problem are in `contended/`.

## Test D: per-stage latency (RUN, numbers pessimistic)

**Setup:**
- `--no-audio`, **paced** playback: frames are dropped when processing falls behind, as a live camera would. Frames are 960 px wide, on an M3 Pro. Script: `bench/latency_runs.sh` and `latency_rerun_gl.sh`.
- **Inputs:**
  - (1) `synthetic_photos.mp4`: the 48 Test B photos, 0.5 s each, with slow pan and jitter, 24 s at 30 fps. There is a target in ~90% of frames and **no hand**.
  - (2) `grasp_lift_trial.ogv`, looped to 25 s: CC BY 4.0, a real hand reaching. Its object is not a cup, so the detector mostly searches.
- **Conditions:** 20 W charger, battery 8%. The probe (`bench/probe.py`, before and after, stable): YOLOE 54 ms, hand 20 ms, Depth Anything 15 ms. YOLOE measured 27 ms at 20:35, so **detection numbers here are about 2× too high**; CPU stages are unaffected. Re-run with the proper charger: `bench/latency_runs.sh && bench/latency_rerun_gl.sh`.

**Real hand in view** (`grasp_lift_trial`), ms median / p95:

| Config | Hand | Detect (when run) | Track | Depth | Guidance | Total / frame | Frame→cue | fps |
|---|---|---|---|---|---|---|---|---|
| Detect every frame, no depth | 20 / 29 | 51 / 69 | — | — | <0.1 | 69 / 86 | 77 / 115 | 16.7 |
| Detect /5 + CSRT, no depth | 20 / 29 | 52 / 70 | see note | — | <0.1 | **38 / 94** | **67 / 116** | 18.1 |
| Detect /5 + CSRT + Depth Anything | 20 / 30 | 52 / 72 | see note | 16 / 30 | <0.1 | **59 / 109** | **89 / 142** | 13.7 |
| Same, Depth Anything every 3rd frame (before `--search-every`) | 20 / 29 | 51 / 69 | 17 / 75 | 0 / 23 | <0.1 | 72 / 105 | 97 / 139 | 13.2 |
| Same + **Cosmos 2B slow loop on the same GPU** | 21 / 30 | 78 / 86 | 10 / 49 | 22 / 47 | <0.1 | 63 / 129 | 92 / 156 | 12.4 |

The grasp clip has no cup, so the target is a false detection that comes and goes. The **track** column there also includes skipped search frames at 0 ms, so use the synthetic table below for tracking cost. These rows ran with `--search-every 2`, so even "detect every frame" skips detection on alternate frames while searching (246 of 418 frames detected).

In the last run, 13 Cosmos calls went through (1.8 s each) with **0 failures**. The fast loop kept going: worst frame 205 ms, the same as without the slow loop. The slowdown comes from sharing the GPU locally, which won't happen with CoreWeave.

**Target in view, tracking working** (synthetic video, no hand):

| Config | Detect | Track | Depth | Total / frame | Frame→cue |
|---|---|---|---|---|---|
| Detect every frame | 77 / 101 | — | — | 89 / 116 | 107 / 135 |
| **Detect /5 + CSRT** | 72 / 86 | 15 / 27 | — | **29 / 90** | 46 / 102 |
| Detect /5 + KCF | 63 / 70 | 2 / 58 | — | **22 / 82** | 31 / 102 |
| Detect /5 + CSRT + Depth Anything | 72 / 85 | 14 / 24 | 17 / 30 | 47 / 111 | 69 / 131 |

**Savings:**
- **Detecting every 5th frame and tracking in between cuts median frame time about 3×** (89 → 29 ms) when a target is being tracked. The p95 stays near the detection cost, because every 5th frame still pays for it.
- **Depth Anything adds ~16–20 ms median.**
- Detection here uses 10 prompts (all 5 categories), which costs more than a single category.
- After seeing this, I made **searching run the detector every 2nd frame** (`--search-every`), because detecting every frame with no target cost 50–80 ms per frame for nothing.

**For the architecture slide**, measured on an M3 Pro and a throttled GPU; expect detection ~2× faster on proper power:

```
Fast loop, per frame (median / p95 ms)       Slow loop, per call
  capture/decode      ~1–5                      Cosmos 8B Q8, local M3 Pro:   5.6–6.9 s
  hand (MediaPipe)    20 / 29                   Cosmos 2B Q8, local M3 Pro:   1.25 s
  detect YOLOE-11s    27 clean (51 throttled), every 5th frame
  track CSRT          8–15 (KCF 2)              Cosmos 8B, H100 vLLM (est.):  ~0.7–1.2 s  <- measure Friday
  depth DAv2-S        16 / 30 (LiDAR: <1)
  guidance            <0.1
  ---------------------------------
  total               38 / 94 (no depth)  59 / 109 (DAv2)
  frame -> cue        67 / 116            89 / 142
  audio: 5.3 ms blocks + 40 ms parameter smoothing (by design, not measured through the DAC)
```

## Test E: open-loop sanity check (PASS, weak evidence)

**Clips** (details in `data/clips/clips_sources.csv`):
- `reach_bottle.ogv`: CC BY 4.0, Koul et al., PLOS ONE 2016. A 1 s side-view reach to a bottle. The target comes from the detector ("bottle").
- `grasp_lift_trial.ogv`: CC BY 4.0, Luciw et al., Scientific Data 2014. A seated reach, frames 200–236. The target box was hand-labelled at frame 200, then tracked with CSRT because the camera reframes during the reach.

**Method:** for each frame, the raw cue direction (dx, dy) is compared with the palm's actual motion over the next 2–3 frames (cosine > 0). Stationary frames are skipped. Both reaches are almost purely horizontal, so I also ran the clips **rotated 90° and 270°**. Those turn them into vertical reaches through the whole hand/detector/guidance pipeline, so the y axis gets tested on real footage. The control flips the sign of `dy`.

| Clip | Scored frames | Agreement | x-sign | y-sign | Control (y flipped) |
|---|---|---|---|---|---|
| bottle, as recorded | 20 | 1.00 | 20/20 | — (0 scored) | 1.00 (y untested) |
| bottle, rotated 90° | 21 | 1.00 | — | 21/21 | **0.00** |
| bottle, rotated 270° | 0 | — | — | — | Detector can't find an upside-down bottle |
| grasp, as recorded | 15 | 1.00 | 15/15 | 3/3 | 1.00 |
| grasp, rotated 90° | 13 | 1.00 | 1/1 | 13/13 | **0.00** |
| grasp, rotated 270° | 17 | 1.00 | 3/3 | 17/17 | **0.00** |

- **86/86 frames agree.** A y-axis sign flip would have dropped the rotated runs to 0%, so the test does catch that bug.
- MediaPipe lost the palm on **13 of 37 frames** of the fast grasp reach (motion blur). The `Guide` holds the last cue for 5 frames, which covers short dropouts.
- **This does not show a blind person can follow the cues.** The people in these clips could see. It only shows that our cue points the way a successful reach goes, so there is no sign or axis bug. It is also only 2 short clips.

---

## Verified vs only built

| Component | Status |
|---|---|
| Guidance maths, dead zone, states, smoothing | **Verified** (unit tests + Test E) |
| Audio synthesis: pan, pitch, rate, click-free | **Verified offline**: unit tests and `output/audio_demo.wav`. **Not heard through speakers.** Listen with `afplay output/audio_demo.wav`. |
| Speech wrapper + gate against Cosmos rewording | Verified with a fake `say`. **Real `say` not run** (no sound at night). |
| Video / image sources, paced dropping, record + replay with depth sidecar | **Verified** |
| Webcam source | **Built only.** It would have triggered the camera permission prompt. |
| Record3D source | **Built only.** Written from the library demo. It fails cleanly with no phone (tested); it has never seen a real frame. Depth-resolution scaling is unit-tested. |
| LiDAR depth sampling | Verified on synthetic depth maps, not on real LiDAR. **Bug found and fixed after the main run:** when the hand covered the target, the target depth sampled the hand, so LiDAR "touching" fired with the hand still in front of the cup. Target depth is now frozen while the palm is over the box (`tests/test_depth_occlusion.py`). |
| Depth Anything V2 Small | Runs; latency measured; the relative sign is sensible on the bottle clip. Not validated against ground truth. |
| Hand tracking | Verified on 2 clips; 65–100% palm found |
| Detector + tracker + Cosmos gating + hold-on-occlusion | Unit-tested with a fake detector; the detector benchmarked (Test B). Hold-on-occlusion was added at the end: unit-tested, seen engaging on the bottle clip, **not tested live**. |
| Slow loop against a real model server | **Verified** with llama.cpp, 2B and 8B: 0 failures in 13 calls while the fast loop ran; failures and timeouts tested against a fake server |
| vLLM on CoreWeave | Not tested: same OpenAI API; `response_format: json_schema` is supported by vLLM |
| `--wandb` | Built; untested (no key). Logging is local CSV/JSON by default. |
| Anthropic API | `ANTHROPIC_API_KEY` was not set, and no LLM calls were needed beyond Cosmos, so it is unused |

## Biggest remaining risk

**Nobody has closed the loop with a person.** Three specific things could sink the live demo, and none of them can be tested from files:

1. **The last 5 cm.**
   - When the hand reaches the object, it covers it. I added "hold the target box while the palm is inside it" so the tracker doesn't latch onto the hand, but that's untested live.
   - Without LiDAR, "touching" means "overlapping in the image". The steady tone can fire with your hand 20 cm in front of the cup.
   - This is why test 2 (LiDAR) matters more than test 1.
2. **Camera placement defines what "left" and "up" mean.** All cues are in image coordinates. They match the user's body only if the camera looks where the user faces: phone on a chest strap or held at the sternum, not on a tripod facing the user. A mirrored or sideways camera silently flips cues. Decide the mount first thing on Friday.
3. **Door hardware depends on Cosmos.** The fast detector finds lever handles about half the time. Cosmos 8B boxes are good (median IoU 0.89), but they are 1–2 s old when they arrive, and CoreWeave latency is unmeasured.

The cheap, useful first hour on Friday: run `bench/cosmos_eval.py` against CoreWeave, then do LIVE_TESTS test 2 with the phone on a chest mount.

## Files

| Path | Contents |
|---|---|
| `handguide/` | The package |
| `tests/` | 55 tests |
| `bench/` | All benchmark scripts |
| `LIVE_TESTS.md`, `RECORDING_CHECKLIST.md`, `README.md` | Docs |
| `data/photos/labels.csv` | Labels + licences |
| `data/clips/clips_sources.csv` | Clip sources |
| `output/bench/` | Raw results |
| `output/runs/latency/*` | Per-frame timing CSVs, cue logs, summaries |
| `output/audio_demo.wav` | Rendered cue demo |
| `output/build_timing_log.md` | Build timing |
| `models/` (13 GB) | Cosmos GGUFs, YOLO weights, text encoders, hand model. Delete `models/Cosmos-*` if you need the space back. |

**Outside this folder:** all models and data are inside `RTHACK/`. A few things live outside it:
- uv's package cache (`~/.cache/uv`).
- The Ultralytics settings file (`~/Library/Application Support/Ultralytics/settings.json`), which the library creates on import.
- Hugging Face's shared Xet chunk cache (`~/.cache/huggingface/xet`), which may still hold ~100 MB of chunks from the first Depth Anything download. I moved the model itself into `models/hf` and deleted its copy in `~/.cache/huggingface/hub`.
- Homebrew `llama.cpp` and `cmake`.
