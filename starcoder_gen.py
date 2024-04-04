"""
Accelerate the generation of StarCoder programs using VLLM.

Usage:
  python starcoder_vllm.py --hf-home={} --hf-cache={} --prompt-dir={} \
  --output-dir={} --n={} --max-tokens={} --split-size={} --log-file={}
"""

import argparse
import time
import os
from pathlib import Path
import logging
from pprint import pprint
import random

from vllm import LLM, SamplingParams

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated-outputs",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="/JawTitan/huggingface/hub",
        help="HuggingFace home dir",
    )
    parser.add_argument(
        "--hf-cache",
        type=str,
        default=None,
        help="HuggingFace cache dir",
    )
    parser.add_argument("-n", "--num", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--split-size", type=int, default=20)
    parser.add_argument("--log-file", type=str, default="whitefox-llm-gen.log")
    parser.add_argument("--model", type=str, default="ise-uiuc/Magicoder-S-DS-6.7B")

    args = parser.parse_args()
    pprint(args)

    if args.hf_home is not None:
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", args.hf_home)

    if args.hf_cache is not None:
        HF_CACHE_DIR = args.hf_cache
    else:
        HF_CACHE_DIR = os.environ.get("HF_HOME", "~/.cache/huggingface")

    logging.basicConfig(level=logging.INFO, filename=args.log_file)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        download_dir=HF_CACHE_DIR,
        max_model_len=16000,
        swap_space=20,
    )

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
    unit_num = 20
    for k in range(0, len(prompts), split_size):
        for j in range(0, n, unit_num):
            cur_num = min(unit_num, n - j)

            st_time = time.time()
            end_idx = min(k + split_size, len(prompts))

            sampling_params = SamplingParams(
                n=cur_num, temperature=1.0, top_p=1.0, max_tokens=max_tokens, stop=EOF_STRINGS, seed=random.randint(0, 10000)
            )

            outputs = llm.generate(prompts[k:end_idx], sampling_params)
            used_time = time.time() - st_time
            logging.info(f"Time taken: {used_time} seconds")

            for i, output in enumerate(outputs):
                filename = filenames[i + k]
                output_file_dir = output_dir / filename
                output_file_dir.mkdir(exist_ok=True, parents=True)
                for r, text in enumerate(output.outputs):
                    generated_text = text.text
                    (output_file_dir / f"{filename}-{j+r}.py").write_text(generated_text)
            (output_dir / f"generated-{k}-{j}-time.log").write_text(str(used_time))


if __name__ == "__main__":
    main()
