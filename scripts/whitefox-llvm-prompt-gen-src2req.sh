#!/bin/bash
set -e

nl_path=${1:-"Requirements-new/llvm/req"}
generate_dir=${2:-"Prompts-generated"}

template_dir="template-llvm"
mode="nl"

curr_dir=$(pwd)
cd ..
# First generate requirements/descriptions using src2nl
for passname in "ADCEPass" "ArgumentPromotionPass" "ConstantMergePass" "DCEPass" "DeadArgumentEliminationPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --mode=src2nl \
        --template=${template_dir}/gpt4-src2mixnl-oneshot.txt \
        --outdir=${generate_dir}/llvm/req2test
done

# Then generate test code using the descriptions
for passname in "ADCEPass" "ArgumentPromotionPass" "ConstantMergePass" "DCEPass" "DeadArgumentEliminationPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --nlpath=${nl_path} \
        --mode=${mode}2test \
        --template=${template_dir}/starcoder-mixnl2test-oneshot.txt \
        --outdir=${generate_dir}/llvm/step1-prompt
done

# Generate feedback-based additional tests
for passname in "ADCEPass" "ArgumentPromotionPass" "ConstantMergePass" "DCEPass" "DeadArgumentEliminationPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --nlpath=${nl_path} \
        --mode=srcnl2test_feedback \
        --template=${template_dir}/starcoder-feedback-mixnlonly.txt \
        --outdir=${generate_dir}/llvm/feedback
done

# Do the same for remaining passes
for passname in "DSEPass" "GlobalDCEPass" "GlobalOptPass" "GVNPass" "InstSimplifyPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --mode=src2nl \
        --template=${template_dir}/gpt4-src2mixnl-oneshot.txt \
        --outdir=${generate_dir}/llvm/req2test
done

# And their tests
for passname in "DSEPass" "GlobalDCEPass" "GlobalOptPass" "GVNPass" "InstSimplifyPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --nlpath=${nl_path} \
        --mode=${mode}2test \
        --template=${template_dir}/starcoder-mixnl2test-oneshot.txt \
        --outdir=${generate_dir}/llvm/step1-prompt
done

# And feedback tests
for passname in "DSEPass" "GlobalDCEPass" "GlobalOptPass" "GVNPass" "InstSimplifyPass"; do
    python3 prompt_gen.py \
        --optpath=llvm-exec/example.json \
        --nlpath=${nl_path} \
        --mode=srcnl2test_feedback \
        --template=${template_dir}/starcoder-feedback-mixnlonly.txt \
        --outdir=${generate_dir}/llvm/feedback
done
cd "$curr_dir"