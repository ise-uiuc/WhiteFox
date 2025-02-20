#!/bin/bash
set -e

# Use single JSON file instead of per-pass files
json_file="llvm-exec/example.json"
generate_dir=${1:-"llvm-exec/Prompts-generated"}
template_dir="llvm-exec/template-llvm"
mode="nl"

python3 gen_prompt_c.py \
    --json=${json_file} \
    --mode=${mode}2test \
    --template=${template_dir}/starcoder-${mode}2test-oneshot-pattern.txt \
    --outdir=${generate_dir}/llvm/step1-prompt