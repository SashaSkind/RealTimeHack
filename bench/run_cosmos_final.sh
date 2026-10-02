#!/bin/bash
# Test C re-run with the final prompt (box before text fields, maxLength in schema).
cd "$(dirname "$0")/.."; source .venv/bin/activate
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag final_8b_schema --json-mode schema
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag final_8b_prompt --json-mode prompt
python bench/cosmos_eval.py --url http://127.0.0.1:8082/v1 --model cosmos-2b --tag final_2b_schema --json-mode schema
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag final_8b_schema_extra --json-mode schema --categories cabinet_handle,light_switch
