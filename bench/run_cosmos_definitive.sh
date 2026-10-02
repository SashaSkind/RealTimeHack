#!/bin/bash
# Definitive Test C numbers: final prompt (box first, schema maxLength, affordance hints), display kept
# awake (caffeinate -u) so the macOS aerial screensaver doesn't steal the GPU.
cd "$(dirname "$0")/.."; source .venv/bin/activate
caffeinate -u -t 3600 &
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag def_8b_schema_hints --json-mode schema
python bench/cosmos_eval.py --url http://127.0.0.1:8082/v1 --model cosmos-2b --tag def_2b_schema_hints --json-mode schema
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag def_8b_schema_nohints --json-mode schema --no-hints
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag def_8b_prompt_hints --json-mode prompt
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag def_8b_schema_hints_extra --json-mode schema --categories cabinet_handle,light_switch
