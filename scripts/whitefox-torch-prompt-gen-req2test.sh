#!/bin/bash
set -e

nl_path=${1:-"Requirements-new/torch-inductor/req"}
generate_dir=${2:-"Prompts-generated"}

template_dir="template-torch"
mode="nl"

for optim_name in 'inductor' 'group-batch'; do
    python prompt_gen.py --optpath=optim/inductor-${optim_name}.json --nlpath=${nl_path} --mode=${mode}2test --template=${template_dir}/starcoder-${mode}2test-oneshot-pattern.txt --outdir=${generate_dir}/torch-inductor/step1-prompt
done

for optim_name in 'mkldnn' 'postgrad' 'sfdp' 'bnfold' 'decompose' 'misc' 'split-cat'; do
    python prompt_gen.py --optpath=optim/inductor-${optim_name}.json --nlpath=${nl_path} --mode=${mode}2test --template=${template_dir}/starcoder-${mode}2test-oneshot-pattern.txt --outdir=${generate_dir}/torch-inductor/step1-prompt
done