#!/bin/bash

generate_dir=${1:-"Prompts-generated"}

template_dir="template-llvm"

# First group of passes
for passname in "ADCEPass" "ArgumentPromotionPass" "ConstantMergePass" "DCEPass" "DeadArgumentEliminationPass"; do
    python3 ../prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --mode=src2nl \
        --template=${template_dir}/gpt4-src2mixnl-oneshot.txt \
        --outdir=${generate_dir}/llvm/req2test
done

# Second group of passes
for passname in "DSEPass" "GlobalDCEPass" "GlobalOptPass" "GVNPass" "InstSimplifyPass"; do
    python3 ../prompt_gen.py \
        --optpath=llvm-exec/example.json  \
        --mode=src2nl \
        --template=${template_dir}/gpt4-src2mixnl-oneshot.txt \
        --outdir=${generate_dir}/llvm/req2test
done