# Recording checklist

These are short clips to record when you have time. Once they exist, every offline test (latency, the open-loop direction check, detector hit rate) can run on your own camera and rooms instead of Commons stand-ins.

Keep each clip **5–15 s**, and put each one in its own folder under `data/recordings/<name>/`. The names below are the ones the scripts expect.

## What to record

| # | Folder name | What happens | Why |
|---|---|---|---|
| 1 | `reach_cup_left` | Cup on a table, in the middle of the frame. Your hand starts at the **left** edge and reaches slowly (about 3 s) to touch the cup. Hold for 2 s. | Checks left/right sign and the `touching` state |
| 2 | `reach_cup_right` | Same, but the hand starts at the **right** edge. | Mirror of #1 |
| 3 | `reach_cup_below` | Same, but the hand starts **below** the cup (bottom of the frame) and moves up. | Checks the up/down sign (image y points down) |
| 4 | `reach_cup_above` | Same, but the hand comes down from **above**. | Mirror of #3 |
| 5 | `reach_cup_depth` | Cup straight ahead, about 60 cm away. The hand starts near the camera and reaches **forward** along the line of sight. | The only clip that exercises depth; must use the iPhone |
| 6 | `reach_lever` | A real door with a lever handle, camera at chest height about 1 m away. Reach, touch, press down. | Detector weak spot; Cosmos box + tracker path |
| 7 | `reach_knob` | Same with a round door knob. | Weakest detector category in Test B |
| 8 | `no_target` | Pan slowly around a room with **no cup or handle** in view, with your hand moving in view. | False positives; the `searching` state |
| 9 | `hand_still` | Cup in view. Hold your open hand still, palm facing the camera, about 50 cm away, for 5 s. | Live test 2 (palm depth stability), offline |
| 10 | `two_hands` | Reach with one hand while the other hand is also in view. | Hand selection logic |
| 11 | `cluttered_cup` | A cup among other objects: bottles, a bowl, a glass. | Detector false positives |

Record each of #1–#5 **twice** if you can: once slow and once at normal speed.

## Recording with the iPhone (RGB + LiDAR). Preferred, because it keeps depth

This streams over USB into our tool, which writes `rgb.mp4` plus `rgb.mp4.depth.npz`. `--source video` replays both.

1. Set up Record3D and USB streaming as in `LIVE_TESTS.md` §0.
2. Mount the phone at chest height in landscape, looking at the scene.
3. In Record3D, press the red button to start streaming. Then on the Mac run:
   ```bash
   source .venv/bin/activate
   python -m handguide.main --source record3d --record data/recordings/reach_cup_left --record-only --max-seconds 15
   ```
   Do the action. Recording stops after 15 s, or when you press Ctrl-C.
4. Check the result: `open data/recordings/reach_cup_left/rgb.mp4`. The file `meta.json` should say `"has_depth": true`.

Frames are resampled to a constant 30 fps by timestamp, so replay timing is honest.

## Recording with a USB webcam (RGB only)

```bash
python -m handguide.main --source webcam --camera-index 0 --record data/recordings/reach_cup_left --record-only --max-seconds 15
```
The first run will trigger the macOS camera permission prompt for your terminal app.

**Fallback with no tool at all:** record with the iPhone Camera app at 1080p/30 and AirDrop the file. Then rename it `data/recordings/<name>/rgb.mp4`. This gives no depth.

## Replaying a recording

```bash
python -m handguide.main --source video --path data/recordings/reach_cup_left/rgb.mp4 --no-audio \
    --target cup --log-dir output/runs/reach_cup_left --save-video output/runs/reach_cup_left/overlay.mp4
```

Replay is **paced** by default: it drops frames if processing falls behind, the way a live camera would. Add `--no-pacing` only for offline analysis where latency doesn't matter. When the clip has a depth sidecar, it is used automatically as LiDAR (`depth_kind=metric`).

To run the direction-agreement check (Test E) on a recording, you need a target box. Either the detector finds it (cups do), or you pass a fixed box:

```bash
python bench/open_loop_check.py --clip data/recordings/reach_cup_left/rgb.mp4 --target cup --approach 0-999
```

`--approach` is the frame range from the start of the reach to contact. Get it from the overlay video, or leave it as the whole clip.
