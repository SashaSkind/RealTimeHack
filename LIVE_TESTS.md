# Live tests: only you can run these

Run them in this order. Each takes 10–20 minutes. All commands run from `~/RTHACK` with the venv active:

```bash
cd ~/RTHACK && source .venv/bin/activate
python -m pytest -q        # expect 51 passed before you start
```

Use **headphones** for the audio: stereo pan is the left/right cue. What the cues mean:

| Cue | Meaning |
|---|---|
| Louder in the right ear | Move right |
| Louder in the left ear | Move left |
| Higher pitch | Move up |
| Lower pitch | Move down |
| Faster beeps | Closer |
| Steady tone | Touching |
| Slow low pulse | Can't see your hand |
| Silence | No target found yet |

To hear it before going live, play the demo, which is rendered by the real synthesiser: `afplay output/audio_demo.wav`.

---

## Seeing the maths live

`--display` opens the camera view with a **maths panel** next to it. It shows each step for every frame:
- palm and target pixels;
- the raw offsets, with the image-y flip;
- the dead zone, then smoothing;
- distance and depth (LiDAR in metres, or relative);
- the audio values: pan, pitch and beeps per second.

Below that are a direction pad, the last 5 s of dx/dy, distance and hand/target depth, and the time each stage takes. `--save-video out.mp4` records the same view, so you can go through it frame by frame afterwards. Add `--no-panel` to hide it.

The terminal also prints a cue line every 15 frames (`--print-every N`; 0 turns it off). Every frame is logged to `<log-dir>/cues.csv` and `<log-dir>/timing.csv`.

## 0. One-time setup

### Camera permission (webcam)
The first run that opens a camera triggers a macOS prompt for your terminal app (Terminal, iTerm or VS Code). Click **Allow**. If you miss it, go to System Settings → Privacy & Security → Camera, enable the terminal app, then restart the terminal.

### iPhone + Record3D (for tests 2 and 3)
1. On the iPhone Pro, install **Record3D — 3D Videos** (by Marek Šimoník) from the App Store.
2. Open it and go to the **Settings** tab. Turn on **"USB Streaming mode"**.
   - If the switch is locked, it will point you to an in-app purchase. As far as I could check tonight, streaming is a paid add-on: a one-time purchase of a few dollars. The Wi-Fi "Extension Pack" is a separate purchase that you **don't** need.
   - I couldn't confirm the exact product name or price from the public pages, so budget 5 minutes for this.
3. Connect the phone to the Mac with a USB cable. Unlock it and tap **Trust This Computer** if asked.
4. In Record3D, choose the **LiDAR (back) camera**, not the selfie TrueDepth camera. Then press the big red button. In USB mode this starts streaming, not recording.
5. On the Mac, check that the phone is seen:
   ```bash
   python -c "from record3d import Record3DStream as R; print(R.get_connected_devices())"
   ```
   You should see one device. If the list is empty: re-plug the cable, make sure the phone is unlocked and the red button is pressed, and close any other program that uses the stream.
6. Mount the phone at chest height, in landscape, pointing at the work area. A cheap phone clamp is enough.

### Cosmos endpoint (for test 3)
- **Friday:** point at the CoreWeave vLLM endpoint:
  ```bash
  export COSMOS_URL=http://<host>:8000/v1 COSMOS_MODEL=<served model name> COSMOS_API_KEY=<key if any>
  ```
  Check the served model name with `curl $COSMOS_URL/models`.
- **Before Friday, locally.** 2B is fast enough to feel live on this Mac (~1.8 s per call); 8B works but takes ~7 s per call:
  ```bash
  llama-server -m models/Cosmos-Reason2-2B-GGUF/Cosmos-Reason2-2B-Q8_0.gguf \
      --mmproj models/Cosmos-Reason2-2B-GGUF/mmproj-Cosmos-Reason2-2B-F16.gguf \
      --port 8082 -ngl 99 -c 8192 --jinja --alias cosmos-2b &
  export COSMOS_URL=http://127.0.0.1:8082/v1 COSMOS_MODEL=cosmos-2b
  ```
  With 2B, add `--no-cosmos-hints` to the commands below: 2B copies the built-in affordance hint instead of looking (Test C).
  **Plug in the real MacBook charger** (≥ 70 W). Tonight's machine was on a 20 W adapter; sustained GPU load drained the battery and throttled everything 2–3×.

---

## Test 1: eyes-closed reach for a cup (webcam + audio)

**Pass:** you touch the cup in **under 10 seconds, three times in a row**, eyes closed, using only the audio. A helper who can see should place the cup and judge the touch. The tool's own "touching" state is logged, but it does not decide pass or fail.

1. Put a mug on a table about arm's length in front of you. Put the webcam about 50 cm behind and above your shoulder, looking down at the table, so both your hand and the cup are in view.
2. Run:
   ```bash
   python -m handguide.main --source webcam --camera-index 0 --target cup --depth off \
       --display --log-dir output/runs/live1_webcam
   ```
   - Leave out `--cosmos-url` here: this test is about the fast loop only.
   - `--depth off` makes "touching" mean overlapping in the image. With a webcam, Depth Anything only gives relative depth. For a second run, try `--depth dav2` and compare.
   - If the wrong camera opens, try `--camera-index 1`.
3. Check the window: a yellow box on the cup and a magenta dot on your palm. Press **r** to arm a trial. Then close your eyes and reach from the side, then from above. The terminal prints `[trial] touching after X s`.
4. Do at least 3 trials. Move the cup between trials. Press **q** to quit.
5. Results:
   - Trial times: `output/runs/live1_webcam/trials.csv`
   - Per-frame cues: `output/runs/live1_webcam/cues.csv`
   - Latency table: printed at exit and saved in `output/runs/live1_webcam/timing.json`

**Also note:**
- Did left/right ever feel backwards? Up/down?
- Were the beeps useful, or annoying?
- Did the steady tone come too early (hand above the cup, not on it)?

The last one is the known weak spot without depth.

---

## Test 2: same with the iPhone over Record3D + palm-depth stability

**Pass (depth):** with your open hand held still about 40–60 cm from the phone, the palm depth varies by less than about **3 cm peak-to-peak** over 2 s.
**Pass (reach):** the same as test 1: three touches in a row, each under 10 s.

1. Set up the phone as in §0. Put the cup about 60 cm in front of the phone.
2. Run:
   ```bash
   python -m handguide.main --source record3d --target cup --depth lidar \
       --display --log-dir output/runs/live2_r3d --record data/recordings/live2_r3d
   ```
   - `--record` also saves the session, with depth, for replay. It costs a little speed. Leave it out if the fps is poor.
   - The overlay text should say `metric h=0.5xx t=0.6xx` (metres).
3. **Depth stability:** hold your open palm still, facing the phone, for 5 s. Then:
   ```bash
   python bench/depth_stability.py output/runs/live2_r3d/cues.csv
   ```
   It prints the quietest 2 s window and PASS or FAIL. If it fails:
   - Look at `hand_depth` in `cues.csv`. Jumps to the table or wall behind the hand mean the sampling window is too big or the palm point is off the hand.
   - Try moving the hand closer.
4. **Reach:** press **r** and do 3 eyes-closed reaches as in test 1. With LiDAR, "touching" also needs palm depth within 6 cm of the cup depth, so an early steady tone should be rarer than in test 1. Note whether that's true.

If `--source record3d` prints `[source] No Record3D device…`, go back through §0 step 5. The rest of the program never crashes on a missing phone; it exits with that message.

---

## Test 3: point at a real door and hear the Cosmos instruction

**Pass:**
- Within about 5 s of pointing at the door, you hear a correct spoken instruction: for a lever, something like "press the handle down"; for a knob, "turn the knob".
- The yellow box lands on the handle. This comes from Cosmos, so the overlay shows the label with source `external`.
- The fast-loop audio keeps running while Cosmos thinks, with no stalls.

1. Stand about 1 m from a door with a lever handle. Hold the phone at chest height (or use the webcam on a laptop).
2. Run:
   ```bash
   python -m handguide.main --source record3d --target "door handle" --goal "open the door" \
       --cosmos-url $COSMOS_URL --cosmos-model $COSMOS_MODEL --cosmos-key "$COSMOS_API_KEY" \
       --slow-interval 1.5 --display --log-dir output/runs/live3_door
   ```
   Use `--source webcam` if the phone isn't set up.
3. Listen for the instruction. It is spoken with macOS `say` and only when it changes. Then reach for the handle using the tones.
4. Repeat with a door knob: `--target "door knob"`.
5. Results:
   - Every Cosmos call, with latency, raw reply and box: `output/runs/live3_door/slow.jsonl`
   - Summary: `slow_calls` / `slow_failures` in `summary.json`

**Watch for:**
- Instructions that repeat word-for-word every 1.5 s. They shouldn't, because repeats are suppressed.
- The box jumping to a different object. The detector is not allowed to override a Cosmos box unless they overlap.
- Stale boxes when you move the camera during a call. The tracker is initialised on the frame Cosmos saw and then moved forward to the current frame; tell me if it still lags.

---

## If something goes wrong

| Symptom | Try |
|---|---|
| No audio | Check the output device in System Settings → Sound. `--no-audio` prints cues instead. |
| Slow fps | Add `--detect-every 10 --depth off`, or `--max-width 640`. |
| Hand not found | More light; keep the palm, not the edge of the hand, toward the camera. |
| Cup not found | Try `--target mug`, or lower `--det-conf 0.05`. |
| Cosmos calls fail | Run `curl $COSMOS_URL/models`. Also try `--json-mode prompt` if the server rejects `response_format`. |
