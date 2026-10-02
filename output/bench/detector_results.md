# Test B: detector benchmark

48 Commons photos, 5 categories (see data/photos/labels.csv). Only the target category's prompts are active per run.
hit = top detection IoU>=0.5; loose = top detection IoU>=0.3 or centre inside label; fp/img = unmatched detections on positive images; neg_fire = share of unrelated-category images where the prompt fires.

## Summary (mean over categories)

| model | prompts@thr | hit | loose | fp/img | neg_fire | mps ms (med/p95) | cpu ms (med/p95) |
|---|---|---|---|---|---|---|---|
| yoloworld | single@0.05 | 0.29 | 0.31 | 0.32 | 0.12 | 25/31 | 63/78 |
| yoloworld | single@0.1 | 0.20 | 0.20 | 0.22 | 0.04 | 25/31 | 63/78 |
| yoloworld | single@0.2 | 0.20 | 0.20 | 0.10 | 0.04 | 25/31 | 63/78 |
| yoloworld | single@0.3 | 0.20 | 0.20 | 0.06 | 0.03 | 25/31 | 63/78 |
| yoloworld | multi@0.05 | 0.39 | 0.41 | 0.60 | 0.20 | 25/31 | 63/78 |
| yoloworld | multi@0.1 | 0.26 | 0.26 | 0.34 | 0.06 | 25/31 | 63/78 |
| yoloworld | multi@0.2 | 0.22 | 0.22 | 0.20 | 0.04 | 25/31 | 63/78 |
| yoloworld | multi@0.3 | 0.20 | 0.20 | 0.14 | 0.03 | 25/31 | 63/78 |
| yoloworld | tuned@0.05 | 0.30 | 0.36 | 0.60 | 0.18 | 25/31 | 63/78 |
| yoloworld | tuned@0.1 | 0.26 | 0.28 | 0.36 | 0.07 | 25/31 | 63/78 |
| yoloworld | tuned@0.2 | 0.22 | 0.24 | 0.20 | 0.05 | 25/31 | 63/78 |
| yoloworld | tuned@0.3 | 0.22 | 0.22 | 0.12 | 0.03 | 25/31 | 63/78 |
| yoloworld-m | single@0.05 | 0.25 | 0.29 | 0.30 | 0.06 | 26/31 | 136/164 |
| yoloworld-m | single@0.1 | 0.20 | 0.24 | 0.22 | 0.03 | 26/31 | 136/164 |
| yoloworld-m | single@0.2 | 0.18 | 0.20 | 0.06 | 0.01 | 26/31 | 136/164 |
| yoloworld-m | single@0.3 | 0.18 | 0.20 | 0.04 | 0.01 | 26/31 | 136/164 |
| yoloworld-m | multi@0.05 | 0.34 | 0.41 | 0.38 | 0.10 | 26/31 | 136/164 |
| yoloworld-m | multi@0.1 | 0.24 | 0.28 | 0.26 | 0.05 | 26/31 | 136/164 |
| yoloworld-m | multi@0.2 | 0.20 | 0.24 | 0.12 | 0.01 | 26/31 | 136/164 |
| yoloworld-m | multi@0.3 | 0.20 | 0.22 | 0.04 | 0.01 | 26/31 | 136/164 |
| yoloworld-m | tuned@0.05 | 0.37 | 0.39 | 0.34 | 0.07 | 26/31 | 136/164 |
| yoloworld-m | tuned@0.1 | 0.24 | 0.26 | 0.24 | 0.02 | 26/31 | 136/164 |
| yoloworld-m | tuned@0.2 | 0.20 | 0.20 | 0.10 | 0.01 | 26/31 | 136/164 |
| yoloworld-m | tuned@0.3 | 0.20 | 0.20 | 0.04 | 0.01 | 26/31 | 136/164 |
| yoloe | single@0.05 | 0.27 | 0.31 | 0.61 | 0.23 | 27/35 | 84/106 |
| yoloe | single@0.1 | 0.22 | 0.24 | 0.39 | 0.11 | 27/35 | 84/106 |
| yoloe | single@0.2 | 0.20 | 0.20 | 0.14 | 0.04 | 27/35 | 84/106 |
| yoloe | single@0.3 | 0.20 | 0.20 | 0.06 | 0.02 | 27/35 | 84/106 |
| yoloe | multi@0.05 | 0.39 | 0.56 | 0.77 | 0.34 | 27/35 | 84/106 |
| yoloe | multi@0.1 | 0.26 | 0.41 | 0.44 | 0.20 | 27/35 | 84/106 |
| yoloe | multi@0.2 | 0.22 | 0.32 | 0.16 | 0.10 | 27/35 | 84/106 |
| yoloe | multi@0.3 | 0.22 | 0.26 | 0.06 | 0.05 | 27/35 | 84/106 |
| yoloe | tuned@0.05 | 0.41 | 0.62 | 0.77 | 0.36 | 27/35 | 84/106 |
| yoloe | tuned@0.1 | 0.32 | 0.43 | 0.46 | 0.22 | 27/35 | 84/106 |
| yoloe | tuned@0.2 | 0.28 | 0.34 | 0.16 | 0.09 | 27/35 | 84/106 |
| yoloe | tuned@0.3 | 0.28 | 0.30 | 0.06 | 0.05 | 27/35 | 84/106 |
| yoloe-m | single@0.05 | 0.29 | 0.39 | 0.58 | 0.20 | 36/42 | 188/229 |
| yoloe-m | single@0.1 | 0.26 | 0.30 | 0.43 | 0.11 | 36/42 | 188/229 |
| yoloe-m | single@0.2 | 0.22 | 0.24 | 0.22 | 0.04 | 36/42 | 188/229 |
| yoloe-m | single@0.3 | 0.22 | 0.24 | 0.12 | 0.01 | 36/42 | 188/229 |
| yoloe-m | multi@0.05 | 0.33 | 0.54 | 0.70 | 0.30 | 36/42 | 188/229 |
| yoloe-m | multi@0.1 | 0.31 | 0.46 | 0.49 | 0.21 | 36/42 | 188/229 |
| yoloe-m | multi@0.2 | 0.28 | 0.37 | 0.22 | 0.09 | 36/42 | 188/229 |
| yoloe-m | multi@0.3 | 0.28 | 0.35 | 0.12 | 0.03 | 36/42 | 188/229 |
| yoloe-m | tuned@0.05 | 0.41 | 0.66 | 0.80 | 0.36 | 36/42 | 188/229 |
| yoloe-m | tuned@0.1 | 0.35 | 0.58 | 0.45 | 0.26 | 36/42 | 188/229 |
| yoloe-m | tuned@0.2 | 0.28 | 0.37 | 0.22 | 0.15 | 36/42 | 188/229 |
| yoloe-m | tuned@0.3 | 0.28 | 0.35 | 0.14 | 0.08 | 36/42 | 188/229 |
| yoloe26 | single@0.05 | 0.28 | 0.28 | 0.30 | 0.15 | 29/35 | 87/112 |
| yoloe26 | single@0.1 | 0.28 | 0.28 | 0.26 | 0.08 | 29/35 | 87/112 |
| yoloe26 | single@0.2 | 0.22 | 0.22 | 0.12 | 0.03 | 29/35 | 87/112 |
| yoloe26 | single@0.3 | 0.22 | 0.22 | 0.06 | 0.01 | 29/35 | 87/112 |
| yoloe26 | multi@0.05 | 0.31 | 0.33 | 0.36 | 0.21 | 29/35 | 87/112 |
| yoloe26 | multi@0.1 | 0.28 | 0.28 | 0.30 | 0.15 | 29/35 | 87/112 |
| yoloe26 | multi@0.2 | 0.26 | 0.26 | 0.12 | 0.07 | 29/35 | 87/112 |
| yoloe26 | multi@0.3 | 0.26 | 0.26 | 0.06 | 0.04 | 29/35 | 87/112 |
| yoloe26 | tuned@0.05 | 0.33 | 0.41 | 0.34 | 0.22 | 29/35 | 87/112 |
| yoloe26 | tuned@0.1 | 0.30 | 0.32 | 0.26 | 0.12 | 29/35 | 87/112 |
| yoloe26 | tuned@0.2 | 0.28 | 0.28 | 0.12 | 0.06 | 29/35 | 87/112 |
| yoloe26 | tuned@0.3 | 0.26 | 0.26 | 0.06 | 0.03 | 29/35 | 87/112 |

## Per category

### yoloworld single@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.10 | 0.14 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.10 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.33 | 0.44 | 0.11 | 0.00 | light_switch_02, light_switch_03, light_switch_04, light_switch_07, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.40 | 0.34 |  |

### yoloworld single@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.11 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.00 | 0.21 |  |

### yoloworld single@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.50 | 0.21 |  |

### yoloworld single@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.16 |  |

### yoloworld multi@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.50 | 0.50 | 0.40 | 0.39 | lever_handle_02, lever_handle_03, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.10 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.44 | 0.56 | 0.11 | 0.10 | light_switch_02, light_switch_03, light_switch_06, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 2.50 | 0.37 |  |

### yoloworld multi@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.30 | 0.30 | 0.20 | 0.11 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.11 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.40 | 0.21 |  |

### yoloworld multi@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.10 | 0.04 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.90 | 0.18 |  |

### yoloworld multi@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.10 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.13 |  |

### yoloworld tuned@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.30 | 0.60 | 0.40 | 0.32 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_08 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.10 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.22 | 0.11 | 0.05 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_06, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 2.50 | 0.37 |  |

### yoloworld tuned@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.30 | 0.20 | 0.14 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.11 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.50 | 0.21 |  |

### yoloworld tuned@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.20 | 0.20 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.80 | 0.18 |  |

### yoloworld tuned@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.16 |  |

### yoloworld-m single@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.33 | 0.44 | 0.00 | 0.08 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_06 |
| cup | 10 | 0.90 | 1.00 | 1.50 | 0.21 |  |

### yoloworld-m single@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.22 | 0.00 | 0.05 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 0.90 | 1.00 | 1.10 | 0.11 |  |

### yoloworld-m single@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 0.90 | 1.00 | 0.30 | 0.05 |  |

### yoloworld-m single@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 0.90 | 1.00 | 0.20 | 0.05 |  |

### yoloworld-m multi@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.50 | 0.50 | 0.20 | 0.11 | lever_handle_01, lever_handle_03, lever_handle_04, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.10 | 0.10 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.44 | 0.00 | 0.10 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_07 |
| cup | 10 | 0.90 | 1.00 | 1.70 | 0.24 |  |

### yoloworld-m multi@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.04 | lever_handle_01, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.10 | 0.10 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.22 | 0.00 | 0.08 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 0.90 | 1.00 | 1.30 | 0.13 |  |

### yoloworld-m multi@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.22 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 0.90 | 1.00 | 0.60 | 0.05 |  |

### yoloworld-m multi@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 0.90 | 1.00 | 0.20 | 0.05 |  |

### yoloworld-m tuned@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.50 | 0.50 | 0.10 | 0.04 | lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.33 | 0.44 | 0.00 | 0.00 | light_switch_02, light_switch_03, light_switch_04, light_switch_06, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.60 | 0.24 |  |

### yoloworld-m tuned@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.22 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.11 |  |

### yoloworld-m tuned@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.50 | 0.05 |  |

### yoloworld-m tuned@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.20 | 0.05 |  |

### yoloe single@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.10 | 0.60 | 0.21 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.40 | 0.14 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.33 | 0.44 | 0.57 | 0.49 | cabinet_handle_01, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.50 | 0.26 |  |

### yoloe single@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.10 | 0.40 | 0.11 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.20 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.14 | 0.21 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.13 |  |

### yoloe single@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.10 | 0.07 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.11 |  |

### yoloe single@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.00 | 0.00 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_06, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.08 |  |

### yoloe multi@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.40 | 0.90 | 0.29 | lever_handle_01, lever_handle_02, lever_handle_04, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.30 | 0.60 | 0.60 | 0.36 | door_knob_01, door_knob_04, door_knob_06, door_knob_07 |
| cabinet_handle | 9 | 0.22 | 0.33 | 0.71 | 0.51 | cabinet_handle_01, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08 |
| light_switch | 9 | 0.22 | 0.44 | 0.11 | 0.26 | light_switch_02, light_switch_03, light_switch_04, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.50 | 0.26 |  |

### yoloe multi@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.30 | 0.50 | 0.18 | lever_handle_01, lever_handle_02, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.10 | 0.40 | 0.20 | 0.25 | door_knob_01, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.29 | 0.31 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.33 | 0.00 | 0.15 | light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.13 |  |

### yoloe multi@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.20 | 0.10 | 0.11 | lever_handle_01, lever_handle_02, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.20 | 0.10 | 0.21 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.22 | 0.00 | 0.05 | light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.11 |  |

### yoloe multi@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.04 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.10 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.11 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.08 |  |

### yoloe tuned@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.50 | 0.60 | 0.80 | 0.43 | lever_handle_01, lever_handle_05, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.20 | 0.60 | 0.70 | 0.36 | door_knob_04, door_knob_06, door_knob_07, door_knob_09 |
| cabinet_handle | 9 | 0.22 | 0.33 | 0.71 | 0.51 | cabinet_handle_01, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08 |
| light_switch | 9 | 0.11 | 0.56 | 0.11 | 0.23 | light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.50 | 0.26 |  |

### yoloe tuned@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.40 | 0.50 | 0.50 | 0.18 | lever_handle_01, lever_handle_04, lever_handle_05, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.10 | 0.30 | 0.30 | 0.36 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.29 | 0.31 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.33 | 0.00 | 0.10 | light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.13 |  |

### yoloe tuned@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.40 | 0.50 | 0.10 | 0.07 | lever_handle_01, lever_handle_04, lever_handle_05, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.10 | 0.21 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.22 | 0.00 | 0.05 | light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.11 |  |

### yoloe tuned@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.40 | 0.40 | 0.00 | 0.04 | lever_handle_01, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.11 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.08 |  |

### yoloe-m single@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.30 | 0.80 | 0.36 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.10 | 0.00 | 0.18 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.22 | 0.29 | 0.21 | cabinet_handle_01, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.33 | 0.00 | 0.05 | light_switch_01, light_switch_02, light_switch_03, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.80 | 0.21 |  |

### yoloe-m single@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.20 | 0.60 | 0.18 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.10 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.14 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.05 | light_switch_01, light_switch_02, light_switch_03, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 1.40 | 0.16 |  |

### yoloe-m single@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.20 | 0.20 | 0.04 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.90 | 0.11 |  |

### yoloe-m single@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.20 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.03 |  |

### yoloe-m multi@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.40 | 1.40 | 0.43 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.20 | 0.00 | 0.21 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.22 | 0.33 | 0.29 | 0.31 | cabinet_handle_01, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.78 | 0.00 | 0.33 | light_switch_03, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.80 | 0.21 |  |

### yoloe-m multi@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.30 | 0.90 | 0.36 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.20 | 0.00 | 0.21 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.14 | 0.08 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.67 | 0.00 | 0.23 | light_switch_02, light_switch_03, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.40 | 0.18 |  |

### yoloe-m multi@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.30 | 0.20 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.56 | 0.00 | 0.13 | light_switch_02, light_switch_03, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 0.90 | 0.11 |  |

### yoloe-m multi@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.30 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.44 | 0.00 | 0.08 | light_switch_01, light_switch_02, light_switch_03, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.03 |  |

### yoloe-m tuned@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.50 | 0.60 | 1.00 | 0.46 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_05 |
| door_knob | 10 | 0.10 | 0.50 | 0.90 | 0.54 | door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_10 |
| cabinet_handle | 9 | 0.22 | 0.33 | 0.29 | 0.28 | cabinet_handle_01, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.89 | 0.00 | 0.31 | light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.80 | 0.21 |  |

### yoloe-m tuned@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.40 | 0.50 | 0.30 | 0.36 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_05, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.40 | 0.40 | 0.43 | door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.14 | 0.08 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.89 | 0.00 | 0.26 | light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 1.40 | 0.18 |  |

### yoloe-m tuned@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.18 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.10 | 0.10 | 0.29 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.56 | 0.00 | 0.13 | light_switch_02, light_switch_03, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 0.90 | 0.11 |  |

### yoloe-m tuned@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.04 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.10 | 0.00 | 0.21 | door_knob_01, door_knob_02, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.00 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.44 | 0.00 | 0.10 | light_switch_01, light_switch_02, light_switch_03, light_switch_06, light_switch_07 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.03 |  |

### yoloe26 single@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.20 | 0.18 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.18 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.10 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.24 |  |

### yoloe26 single@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.20 | 0.07 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.08 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.00 | 0.16 |  |

### yoloe26 single@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.08 |  |

### yoloe26 single@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.10 | 0.10 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.00 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.00 | 0.00 | 0.00 | 0.00 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08, light_switch_09 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.03 |  |

### yoloe26 multi@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.50 | 0.29 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.31 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.33 | 0.00 | 0.15 | light_switch_01, light_switch_03, light_switch_04, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.24 |  |

### yoloe26 multi@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.10 | 0.07 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.40 | 0.25 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.15 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.10 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.00 | 0.16 |  |

### yoloe26 multi@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.00 | 0.04 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.14 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.08 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.08 |  |

### yoloe26 multi@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.00 | 0.00 | lever_handle_01, lever_handle_02, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.03 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.03 |  |

### yoloe26 tuned@0.05

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.30 | 0.50 | 0.20 | 0.18 | lever_handle_03, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.30 | 0.25 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.23 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.22 | 0.44 | 0.00 | 0.18 | light_switch_01, light_switch_04, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.20 | 0.24 |  |

### yoloe26 tuned@0.1

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.30 | 0.40 | 0.10 | 0.07 | lever_handle_01, lever_handle_03, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.20 | 0.18 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.11 | 0.11 | 0.00 | 0.08 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.10 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 1.00 | 0.16 |  |

### yoloe26 tuned@0.2

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.30 | 0.30 | 0.00 | 0.00 | lever_handle_01, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.11 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.05 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.08 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 0.60 | 0.08 |  |

### yoloe26 tuned@0.3

| category | n | hit | loose | fp/img | neg_fire | loose misses |
|---|---|---|---|---|---|---|
| lever_handle | 10 | 0.20 | 0.20 | 0.00 | 0.00 | lever_handle_01, lever_handle_03, lever_handle_04, lever_handle_05, lever_handle_07, lever_handle_08, lever_handle_09, lever_handle_10 |
| door_knob | 10 | 0.00 | 0.00 | 0.00 | 0.04 | door_knob_01, door_knob_02, door_knob_03, door_knob_04, door_knob_05, door_knob_06, door_knob_07, door_knob_08, door_knob_09, door_knob_10 |
| cabinet_handle | 9 | 0.00 | 0.00 | 0.00 | 0.03 | cabinet_handle_01, cabinet_handle_02, cabinet_handle_03, cabinet_handle_04, cabinet_handle_05, cabinet_handle_06, cabinet_handle_07, cabinet_handle_08, cabinet_handle_09 |
| light_switch | 9 | 0.11 | 0.11 | 0.00 | 0.05 | light_switch_01, light_switch_02, light_switch_03, light_switch_04, light_switch_05, light_switch_06, light_switch_07, light_switch_08 |
| cup | 10 | 1.00 | 1.00 | 0.30 | 0.03 |  |
