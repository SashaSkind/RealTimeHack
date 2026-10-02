"""Wires the fast loop (every frame) and the slow loop (Cosmos, every 1-2 s) together.

Examples:
  python -m handguide.main --source video --path clip.mp4 --no-audio --target cup
  python -m handguide.main --source webcam --target cup --cosmos-url http://HOST:8000/v1
  python -m handguide.main --source record3d --target "door handle" --goal "open the door"
  python -m handguide.main --source record3d --record data/recordings/reach_cup_left --record-only
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .audio import AudioConfig, AudioOut, CueLogger
from .debugviz import DebugPanel
from .depth import DepthEstimator
from .guidance import Guide, GuidanceConfig
from .hand import HandTracker
from .slow_loop import CosmosClient, LatestFrame, SlowLoop
from .sources import RotatedSource, SourceError, open_source
from .speech import SpeechGate, Speaker
from .target import OpenVocabDetector, TargetTracker
from .timing import StageTimer

STATE_COLORS = {"searching": (128, 128, 128), "lost": (255, 0, 0), "guiding": (255, 200, 0),
                "close": (0, 200, 255), "touching": (0, 255, 0)}


def draw_overlay(rgb, hand, tstate, cue, depth, slow=None):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    col = STATE_COLORS.get(cue.state, (255, 255, 255))[::-1]
    if tstate.box is not None:
        x1, y1, x2, y2 = map(int, tstate.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, f"{tstate.label} {tstate.source}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if hand.found:
        cv2.circle(img, (int(hand.x), int(hand.y)), 8, (255, 0, 255), -1)
        if tstate.box is not None:
            tx, ty = (tstate.box[0] + tstate.box[2]) / 2, (tstate.box[1] + tstate.box[3]) / 2
            cv2.arrowedLine(img, (int(hand.x), int(hand.y)), (int(tx), int(ty)), col, 2, tipLength=0.05)
    txt = f"{cue.state} dx={cue.dx:+.2f} dy={cue.dy:+.2f} d={cue.distance:.2f}"
    if depth.kind != "none":
        txt += f" {depth.kind} h={depth.hand if depth.hand is None else round(depth.hand, 3)} t={depth.target if depth.target is None else round(depth.target, 3)}"
    cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    if slow is not None:
        cv2.putText(img, f"cosmos: {slow.target} | {slow.affordance}"[:90], (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


class Recorder:
    """Writes raw frames at a constant frame rate (duplicating/dropping by timestamp) to
    DIR/rgb.mp4, and LiDAR depth to DIR/rgb.mp4.depth.npz, so `--source video` replays it."""

    def __init__(self, out_dir, fps=30.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.depths, self.ts = [], []
        self.next_due = None
        self.t0 = None

    def add(self, frame):
        if self.writer is None:
            h, w = frame.rgb.shape[:2]
            self.writer = cv2.VideoWriter(str(self.dir / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t0 = frame.ts
            self.next_due = frame.ts
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        while self.next_due <= frame.ts + 1e-6:
            self.writer.write(bgr)
            self.ts.append(self.next_due - self.t0)
            if frame.depth is not None:
                self.depths.append(frame.depth.astype(np.float16))
            self.next_due += 1.0 / self.fps

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.depths:
            np.savez_compressed(self.dir / "rgb.mp4.depth.npz", depth=np.stack(self.depths), ts=np.array(self.ts))
        (self.dir / "meta.json").write_text(json.dumps({"fps": self.fps, "frames": len(self.ts),
                                                         "has_depth": bool(self.depths)}, indent=2))


class TrialTimer:
    """Times a reach: starts at the first guiding/close frame after a reset, stops at the first
    (debounced) touching frame. Press `r` in the --display window to arm the next trial.
    The live pass/fail is still judged by a person watching; this just records the times."""

    def __init__(self, path):
        self.path = Path(path)
        self.times: list[float] = []
        self.reset()

    def reset(self):
        self.t0 = None
        self.done = False
        print("[trial] armed")

    def update(self, state, speaker=None):
        if self.done:
            return
        if self.t0 is None and state in ("guiding", "close"):
            self.t0 = time.time()
        elif self.t0 is not None and state == "touching":
            dt = time.time() - self.t0
            self.times.append(round(dt, 2))
            self.done = True
            print(f"[trial] touching after {dt:.1f} s (trial {len(self.times)}); press r to arm the next one")
            with open(self.path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')},{dt:.2f}\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hand guidance: fast loop + Cosmos slow loop")
    p.add_argument("--source", choices=["webcam", "video", "images", "record3d", "pathon"], default="webcam")
    p.add_argument("--path", help="video file or image folder (for --source pathon: iPhone IP for Wi-Fi; omit for USB)")
    p.add_argument("--camera-index", type=int, default=0, help="webcam index or Record3D device index")
    p.add_argument("--no-pacing", action="store_true", help="video/images: process every frame as fast as possible (offline only; latency numbers will be optimistic)")
    p.add_argument("--loop", action="store_true", help="loop the video")
    p.add_argument("--repeat", type=int, default=1, help="images: yield each image N times")
    p.add_argument("--max-width", type=int, default=960, help="downscale frames wider than this")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate frames clockwise (phone held upright sends sideways frames: try 90 or 270)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0)
    # target
    p.add_argument("--target", default="cup", help="initial detector prompt / category")
    p.add_argument("--detector", default="yoloe", help="yoloe | yoloe-m | yoloe26 | yoloworld | yoloworld-m | none")
    p.add_argument("--det-conf", type=float, default=0.1)
    p.add_argument("--det-imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="torch device for detector: mps | cpu (default: ultralytics picks)")
    p.add_argument("--detect-every", type=int, default=5)
    p.add_argument("--tracker", default="csrt", choices=["csrt", "kcf", "mil"])
    p.add_argument("--search-every", type=int, default=2, help="with no target, run the detector every Nth frame")
    p.add_argument("--no-hold", action="store_true", help="keep tracking even while the palm is over the target")
    p.add_argument("--track-scale", type=float, default=0.5)
    # depth
    p.add_argument("--depth", default="auto", choices=["auto", "lidar", "dav2", "off"])
    p.add_argument("--depth-every", type=int, default=1)
    p.add_argument("--depth-size", type=int, default=252)
    # output
    p.add_argument("--no-audio", action="store_true", help="log cue values instead of playing audio or speech")
    p.add_argument("--no-speech", action="store_true")
    p.add_argument("--voice", default=None)
    p.add_argument("--display", action="store_true", help="show an OpenCV window with the overlay")
    p.add_argument("--save-video", help="write the annotated overlay to this mp4")
    p.add_argument("--no-panel", action="store_true", help="hide the live maths/debug panel next to the overlay")
    p.add_argument("--record", help="save raw RGB (+LiDAR depth) to this folder for later replay")
    p.add_argument("--record-only", action="store_true", help="only record, skip processing")
    p.add_argument("--log-dir", default="output/runs/latest")
    p.add_argument("--print-every", type=int, default=15)
    p.add_argument("--wandb", action="store_true", help="also log the summary to Weights & Biases (optional)")
    # slow loop
    p.add_argument("--cosmos-url", default=os.environ.get("COSMOS_URL"), help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8082/v1")
    p.add_argument("--cosmos-model", default=os.environ.get("COSMOS_MODEL", "cosmos"))
    p.add_argument("--cosmos-key", default=os.environ.get("COSMOS_API_KEY", "none"))
    p.add_argument("--goal", default=None, help="what the user wants, e.g. 'open the door' (default: 'pick up the <target>')")
    p.add_argument("--slow-interval", type=float, default=1.5)
    p.add_argument("--cosmos-timeout", type=float, default=20.0)
    p.add_argument("--box-format", default="norm1000", choices=["norm1000", "pixel"])
    p.add_argument("--json-mode", default="schema", choices=["schema", "object", "prompt"])
    p.add_argument("--no-cosmos-hints", action="store_true",
                   help="drop the affordance hint sentence. Use with Cosmos 2B: it parrots the first hint (Test C)")
    return p.parse_args(argv)


def run(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = open_source(args.source, args.path, paced=not args.no_pacing, camera_index=args.camera_index,
                          repeat=args.repeat, max_width=args.max_width, loop=args.loop)
    except SourceError as e:
        print(f"[source] {e}", file=sys.stderr)
        return 2
    if args.rotate:
        src = RotatedSource(src, args.rotate)

    recorder = Recorder(args.record) if args.record else None
    if args.record_only:
        print(f"[record] recording to {args.record}. Ctrl-C to stop.")
        t0 = time.time()
        try:
            for fr in src:
                recorder.add(fr)
                if args.max_seconds and time.time() - t0 > args.max_seconds:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            recorder.close()
            src.close()
        print(f"[record] wrote {len(recorder.ts)} frames to {args.record}")
        return 0

    hand = HandTracker(mode="video")
    detector = None if args.detector == "none" else OpenVocabDetector(args.detector, device=args.device,
                                                                         conf=args.det_conf, imgsz=args.det_imgsz)
    target = TargetTracker(detector, target=args.target, detect_every=args.detect_every,
                           tracker=args.tracker, track_scale=args.track_scale, search_every=args.search_every)
    depth = DepthEstimator(args.depth, every=args.depth_every, input_size=args.depth_size)
    guide = Guide(GuidanceConfig())
    logger = CueLogger(log_dir / "cues.csv", print_every=args.print_every)
    audio = None if args.no_audio else AudioOut(AudioConfig())
    speaker = Speaker(voice=args.voice, enabled=not (args.no_audio or args.no_speech))
    timer = StageTimer()

    frames_box = LatestFrame()
    slow = None
    slow_log = open(log_dir / "slow.jsonl", "w")
    gate = SpeechGate()

    def on_result(res):
        if res.box is not None:
            target.set_external_box(res.box, label=None, frame_rgb=res.frame_rgb)
        if gate.should_speak(res.instruction, res.phase):
            speaker.speak(res.instruction)

    if args.cosmos_url:
        goal = args.goal or f"pick up the {args.target}"
        client = CosmosClient(args.cosmos_url, model=args.cosmos_model, api_key=args.cosmos_key,
                              timeout_s=args.cosmos_timeout, box_format=args.box_format, json_mode=args.json_mode,
                              hints=not args.no_cosmos_hints)
        slow = SlowLoop(client, frames_box, goal, interval_s=args.slow_interval, on_result=on_result)
        slow.start()

    panel = None if args.no_panel else DebugPanel()
    trial = TrialTimer(log_dir / "trials.csv")
    writer = None
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("flag", True))

    # Warm-up so model load time doesn't pollute the latency numbers.
    first = True
    t_start = time.time()
    n = 0
    try:
        it = iter(src)
        while not stop["flag"]:
            t_wait = time.perf_counter()
            try:
                fr = next(it)
            except StopIteration:
                break
            except SourceError as e:  # e.g. the phone stopped streaming: end the run but keep the logs
                print(f"[source] {e} -- stopping and saving the run", file=sys.stderr)
                break
            if first:
                hand.detect(fr.rgb, int(fr.ts * 1000))
                target.update(fr.rgb)
                depth.update(fr.rgb, fr.depth, None, None)
                if depth.mode in ("auto", "dav2") and fr.depth is None:
                    depth._get_dav2()(fr.rgb)
                first = False
                timer.rows.clear()
                src.stats.dropped = 0
                if hasattr(src, "reset_clock"):
                    src.reset_clock()
                t_start = time.time()
                continue
            timer.start_frame(frame=fr.index)
            timer.add("capture", src.stats.last_capture_ms)
            frames_box.put(fr)
            if recorder:
                recorder.add(fr)

            tb_prev = target.state.box
            prefer = None if tb_prev is None else ((tb_prev[0] + tb_prev[2]) / 2, (tb_prev[1] + tb_prev[3]) / 2)
            with timer.stage("hand"):
                h = hand.detect(fr.rgb, int(fr.ts * 1000), prefer_near=prefer)
            hold = False
            if h.found and tb_prev is not None:  # palm inside the (slightly grown) target box
                mx, my = 0.15 * (tb_prev[2] - tb_prev[0]), 0.15 * (tb_prev[3] - tb_prev[1])
                hold = (tb_prev[0] - mx <= h.x <= tb_prev[2] + mx) and (tb_prev[1] - my <= h.y <= tb_prev[3] + my)
            t = time.perf_counter()
            ts = target.update(fr.rgb, hold=hold and not args.no_hold)
            dt = (time.perf_counter() - t) * 1000
            timer.add("target", dt)
            timer.add("detect" if "detect" in ts.op else ("track" if ts.op == "track" else ts.op if ts.op == "hold" else "idle"), dt)
            with timer.stage("depth"):
                d = depth.update(fr.rgb, fr.depth, h.xy, ts.box, h.palm_size, freeze_target=hold)
            with timer.stage("guidance"):
                cue = guide.update(h.xy, ts.box, (fr.rgb.shape[1], fr.rgb.shape[0]), d.hand, d.target, d.kind)
            with timer.stage("output"):
                logger.update(cue, t=fr.ts, frame=fr.index, hand_depth=d.hand, target_depth=d.target,
                              hand_xy=h.xy, target_box=ts.box)
                trial.update(cue.state, speaker)
                if audio:
                    audio.update(cue)
            latency = (time.perf_counter() - fr.due_wall) * 1000 if fr.due_wall else None
            timer.end_frame(latency=latency, target_op=ts.op, dav2_ran=d.ran_model, hand_found=h.found,
                            has_target=ts.box is not None, state=cue.state)

            if args.display or args.save_video:
                img = draw_overlay(fr.rgb, h, ts, cue, d, slow.latest() if slow else None)
                if panel is not None:
                    el = time.time() - t_start
                    pimg = panel.render(img.shape[0], (fr.rgb.shape[1], fr.rgb.shape[0]), fr.ts, h, ts, cue, d,
                                        timer.rows[-1] if timer.rows else None, fps=(n + 1) / el if el > 0 else None)
                    if pimg.shape[0] > img.shape[0]:
                        img = cv2.copyMakeBorder(img, 0, pimg.shape[0] - img.shape[0], 0, 0,
                                                 cv2.BORDER_CONSTANT, value=(25, 26, 26))
                    img = np.hstack([img, pimg])
                if args.save_video:
                    if writer is None:
                        fps = getattr(src, "fps", 30.0)
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
                    writer.write(img)
                if args.display:
                    cv2.imshow("handguide", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        trial.reset()
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
            if args.max_seconds and time.time() - t_start > args.max_seconds:
                break
    finally:
        if slow:
            slow.stop()
        src.close()
        logger.close()
        if audio:
            audio.close()
        speaker.close()
        if writer:
            writer.release()
        if recorder:
            recorder.close()
        if args.display:
            cv2.destroyAllWindows()
        if slow:
            for r in slow.history:
                slow_log.write(json.dumps({k: v for k, v in r.__dict__.items() if k != "frame_rgb"}) + "\n")
        slow_log.close()

    wall = time.time() - t_start
    timer.save(log_dir / "timing")
    stats = src.stats
    summary = {
        "frames_processed": n, "wall_s": round(wall, 2), "processed_fps": round(n / wall, 2) if wall else 0,
        "source_read": stats.read, "source_dropped": stats.dropped, "paced": not args.no_pacing,
        "detector": args.detector, "detect_every": args.detect_every, "depth": args.depth,
        "stages": timer.summary(),
        "slow_calls": slow.calls if slow else 0, "slow_failures": slow.failures if slow else 0,
        "spoken": [t for _, t in speaker.spoken],
        "trials_s": trial.times,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(timer.summary_table())
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, indent=2))
    if args.wandb:
        try:
            import wandb
            wandb.init(project="handguide", config=vars(args))
            wandb.log({f"{k}_{m}": v[m] for k, v in summary["stages"].items() for m in ("median_ms", "p95_ms")})
            wandb.finish()
        except Exception as e:
            print(f"[wandb] skipped: {e}")
    return 0


def main(argv=None):
    sys.exit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
