#!/bin/bash
# Test E on both CC BY clips, as recorded and rotated 90/270 degrees (rotation turns the horizontal reaches vertical).
cd "$(dirname "$0")/.."; source .venv/bin/activate
for r in 0 90 270; do
  python bench/open_loop_check.py --clip data/clips/grasp_lift_trial.ogv --box 780,530,850,645@200 --approach 200-236 --rotate $r >/dev/null 2>&1
  python bench/open_loop_check.py --clip data/clips/reach_bottle.ogv --target bottle --k 2 --rotate $r >/dev/null 2>&1
done
python - <<'PY'
import json, glob
for f in sorted(glob.glob("output/bench/open_loop/*/summary.json")):
    d = json.load(open(f))
    keep = ("window_frames","hand_found_in_window","target_found_in_window","scored_frames","agreement","agreement_smoothed_cue","x_sign_agreement","x_n","y_sign_agreement","y_n","control_agreement_if_y_flipped")
    print(f.split("/")[-2], {k: (round(d[k], 2) if isinstance(d[k], float) else d[k]) for k in keep})
PY
