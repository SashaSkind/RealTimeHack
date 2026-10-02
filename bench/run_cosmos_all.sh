#!/bin/bash
# Test C runs, sequential so GPU timings don't contend. Needs llama-server on 8082 (2B) and 8083 (8B).
set -x
cd "$(dirname "$0")/.."
source .venv/bin/activate
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag 8b_prompt_norm1000 --json-mode prompt
python bench/cosmos_eval.py --url http://127.0.0.1:8082/v1 --model cosmos-2b --tag 2b_prompt_norm1000 --json-mode prompt
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag 8b_schema_norm1000 --json-mode schema
python bench/cosmos_eval.py --url http://127.0.0.1:8083/v1 --model cosmos-8b --tag 8b_prompt_pixel --json-mode prompt --box-format pixel
