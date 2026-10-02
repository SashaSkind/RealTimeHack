#!/bin/bash
# Test D follow-up: grasp-clip runs after adding --search-every, plus the slow loop running concurrently.
cd "$(dirname "$0")/.."; source .venv/bin/activate
GL="--source video --path data/clips/grasp_lift_trial.ogv --loop --max-seconds 25 --target cup --no-audio --print-every 0"
run() { name=$1; shift; echo "== $name"; python -m handguide.main $GL "$@" --log-dir output/runs/latency/$name 2>&1 | grep -E "^\| (total|latency|hand|target|detect|track|depth) |processed_fps|slow_calls|slow_failures"; }
run gl_det5_nodepth --detect-every 5 --depth off
run gl_det5_dav2 --detect-every 5 --depth dav2
run gl_det1_nodepth --detect-every 1 --depth off
if curl -s localhost:8082/health >/dev/null; then
  run gl_det5_dav2_slow2b --detect-every 5 --depth dav2 --cosmos-url http://127.0.0.1:8082/v1 --cosmos-model cosmos-2b --no-cosmos-hints --goal "pick up the object on the stand"
fi
