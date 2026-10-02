#!/bin/bash
# Test D: per-stage latency of the fast loop, paced (real-time) playback, --no-audio.
# Run with nothing else using the GPU (stop llama-server unless a run needs it).
cd "$(dirname "$0")/.."
source .venv/bin/activate
ALL="cup,door handle,door knob,cabinet handle,light switch"
run() { name=$1; shift; echo "== $name"; python -m handguide.main --no-audio --print-every 0 --log-dir output/runs/latency/$name "$@" 2>&1 | grep -E "^\||processed_fps|source_dropped|frames_processed"; }
SYN="--source video --path data/clips/synthetic_photos.mp4"
GL="--source video --path data/clips/grasp_lift_trial.ogv --loop --max-seconds 25 --target cup"
run syn_det1_nodepth   $SYN --target "$ALL" --detect-every 1 --depth off
run syn_det5_nodepth   $SYN --target "$ALL" --detect-every 5 --depth off
run syn_det5_dav2      $SYN --target "$ALL" --detect-every 5 --depth dav2
run syn_det5_kcf       $SYN --target "$ALL" --detect-every 5 --depth off --tracker kcf
run gl_det1_nodepth    $GL --detect-every 1 --depth off
run gl_det5_nodepth    $GL --detect-every 5 --depth off
run gl_det5_dav2       $GL --detect-every 5 --depth dav2
run gl_det5_dav2_every3 $GL --detect-every 5 --depth dav2 --depth-every 3
run gl_det5_kcf_dav2   $GL --detect-every 5 --depth dav2 --tracker kcf
if curl -s localhost:8082/health >/dev/null; then
  run gl_det5_dav2_slow2b $GL --detect-every 5 --depth dav2 --cosmos-url http://127.0.0.1:8082/v1 --cosmos-model cosmos-2b --goal "pick up the cup"
fi
