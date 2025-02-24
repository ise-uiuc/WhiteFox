"""
Accelerate the generation of StarCoder programs using Ollama.

Usage:
  python starcoder_ollama.py --prompt-dir={} --output-dir={} --n={} \
  --max-tokens={} --split-size={} --log-file={}
"""

import argparse
import time
import os
from pathlib import Path
import logging
from pprint import pprint
import random
import requests
import json

EOF_STRINGS = [
    "<|endoftext|>",
    "###",
    "__output__ =",
    "if __name__",
    '"""',
    "'''",
    "# Model ends",
    "# LLVM IR ends",
    "# C++ Code ends",
]

def generate_with_ollama(prompt, model="starcoder", max_tokens=4096, temperature=1.0, top_p=1.0):
    """Generate text using Ollama API"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": EOF_STRINGS
        },
        "stream": False
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API error: {response.status_code}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated-outputs",
    )
    parser.add_argument("-n", "--num", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--split-size", type=int, default=20)
    parser.add_argument("--log-file", type=str, default="whitefox-llm-gen.log")
    parser.add_argument("--model", type=str, default="starcoder")

    args = parser.parse_args()
    pprint(args)

    logging.basicConfig(level=logging.INFO, filename=args.log_file)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    prompts = []
    filenames = []
    for prompt_file in prompt_dir.glob("*.txt"):
        with open(prompt_file) as f:
            prompts.append(f.read())
        filenames.append(prompt_file.stem)
    logging.info(f"Number of prompts: {len(prompts)}")
    print(f"Number of prompts: {len(prompts)}")

    n = args.num
    max_tokens = args.max_tokens
    split_size = args.split_size
    
    for k in range(0, len(prompts), split_size):
        for j in range(n):
            st_time = time.time()
            end_idx = min(k + split_size, len(prompts))
            
            for i in range(k, end_idx):
                try:
                    generated_text = generate_with_ollama(
                        prompts[i],
                        model=args.model,
                        max_tokens=max_tokens,
                        temperature=1.0,
                        top_p=1.0
                    )
                    
                    filename = filenames[i]
                    output_file_dir = output_dir / filename
                    output_file_dir.mkdir(exist_ok=True, parents=True)
                    
                    output_file = output_file_dir / f"{filename}-{j}.py"
                    output_file.write_text(generated_text)
                    
                except Exception as e:
                    logging.error(f"Error generating for {filename}: {str(e)}")
                    continue
            
            used_time = time.time() - st_time
            logging.info(f"Time taken: {used_time} seconds")
            (output_dir / f"generated-{k}-{j}-time.log").write_text(str(used_time))

if __name__ == "__main__":
    main()