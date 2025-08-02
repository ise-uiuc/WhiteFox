"""
Check whether all Ollama generation tasks are done.

Usage:
```
python ollama_call.py --prompt-dir=/path/to/prompts/{target}/{step} --outdir=/path/to/ollama-output
```
"""
from pathlib import Path
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompt/pytorch/step0")
    parser.add_argument("--outdir", type=str, default="ollama-generated")
    parser.add_argument("--sleep-time", type=int, default=30)
    parser.add_argument("--num", type=int, default=10,
                        help="Expected number of generations per prompt")

    args = parser.parse_args()
    prompt_dir = Path(args.prompt_dir)
    outdir = Path(args.outdir)

    prompt_name = prompt_dir.stem
    while True:
        all_done = True
        not_done_cnt = 0
        for prompt_file in prompt_dir.iterdir():
            if not prompt_file.is_file(): continue
            if prompt_file.suffix != ".txt": continue

            out_file = (outdir / prompt_file.stem / f"{prompt_file.stem}_{args.num}.py")
            if not out_file.exists():
                all_done = False
                not_done_cnt += 1
            else:
                print(f"{prompt_file.stem} is done")
        if all_done:
            print("All Ollama generation tasks completed!")
            break
        print(f"Waiting for {not_done_cnt} prompts to be processed by Ollama...")
        time.sleep(args.sleep_time)
