#!/bin/bash

generate_dir=${1:-"Prompts-generated"}

template_dir="llvm-prompts"

# Generate prompts for the NL generation.
for optim_name: 
    python3 ollama.py --optpath=example.json --template=${template_dir}/starcoder-src2nl.txt --outdir=${generate_dir}/llvm-prompts/req2test
done
