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
        default=None,
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
    parser.add_argument("--log-file", type=str, default="starcoder.log")

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
        model="bigcode/starcoderbase",
        dtype="bfloat16",
        download_dir=HF_CACHE_DIR,
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

    sampling_params = SamplingParams(
        n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens, stop=EOF_STRINGS
    )

    split_size = args.split_size
    for k in range(0, len(prompts), split_size):
        st_time = time.time()
        end_idx = min(k + split_size, len(prompts))
        outputs = llm.generate(prompts[k:end_idx], sampling_params)
        used_time = time.time() - st_time
        logging.info(f"Time taken: {used_time} seconds")

        for i, output in enumerate(outputs):
            filename = filenames[i + k]
            output_file_dir = output_dir / filename
            output_file_dir.mkdir(exist_ok=True, parents=True)
            for j, text in enumerate(output.outputs):
                generated_text = text.text
                (output_file_dir / f"{filename}-{j}.py").write_text(generated_text)
        (output_dir / f"generated-{k}-time.log").write_text(str(used_time))


if __name__ == "__main__":
    main()
