"""
The script checks whether all generated tasks are done.

Usage:
```
python starcoder_call.py  --prompt-dir=/JawTitan/whitefox-data/prompts/{target_name}/{step_name} --outdir=/JawTitan/whitefox-data/starcoder
```
"""
from pathlib import Path
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompt/pytorch/step0")
    parser.add_argument("--outdir", type=str, default="starcoder")
    parser.add_argument("--sleep-time", type=int, default=30)

    args = parser.parse_args()
    prompt_dir = Path(args.prompt_dir)
    outdir = Path(args.outdir)

    # target_name = prompt_dir.parent.stem
    prompt_name = prompt_dir.stem
    # Tell whether all the prompts are generated
    while True:
        all_done = True
        not_done_cnt = 0
        for prompt_file in prompt_dir.iterdir():
            if not prompt_file.is_file(): continue

            out_file = (outdir / prompt_file.stem / f"{prompt_file.stem}_10.py")
            if not out_file.exists():
                all_done = False
                not_done_cnt += 1
            else:
                print(f"{prompt_file.stem} is done")
        if all_done:
            break
        print(f"Waiting for {not_done_cnt} optims to be done...")
        time.sleep(args.sleep_time)
