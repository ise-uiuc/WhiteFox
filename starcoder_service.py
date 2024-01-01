"""
The service backend for starcode.

Please use the following command to start the service:
```
python starcoder_service.py --prompt-dir=/JawTitan/whitefox-data/prompts --output-dir=/JawTitan/whitefox-data/starcoder --device='cuda:N' --num=10
```
where N is the GPU device number.

If you have prompts to be used to generate the code, please put them in the `/JawTitan/whitefox-data/prompts/{target_name}/{step_name}/{prompts}`. The service will scan the prompt-dir every 30 seconds and generate code for the new prompts.
For example, `/JawTitan/whitefox-data/prompts/pytorch-inductor/step0/{prompts}` contains the prompts for the first step of the pytorch inductor.

The output of the prompts will be put in `/JawTitan/whitefox-data/starcode/{target_name}/{step_name}/{prompt_name}/{generated}`. For example, `/JawTitan/whitefox-data/starcode/pytorch-inductor/step0/hello/{generated}` contains the generated code for `hello` optimization in the first step of the pytorch inductor.

For the log file, you can find it in `/JawTitan/whitefox-data/prompts/log.txt`.
"""

import torch
import argparse
from datetime import datetime
import time
from math import ceil
import os
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


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


class Logger:
    def __init__(self, log_file: Path, is_print=False) -> None:
        self.log_file = log_file
        self.is_print = is_print

        # Initialize log file.
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        with open(self.log_file, "a") as f:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            f.write("====================\n")
            f.write(f"[{formatted_datetime}] Start logging.\n")

    def log(self, msg):
        if self.is_print:
            print(msg)
        timestamp = datetime.now().strftime("%d.%b %Y %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class StarCoder:
    def __init__(self, device="cuda", max_length=8192) -> None:
        checkpoint = "bigcode/starcoder"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, cache_dir=HF_CACHE_DIR
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=HF_CACHE_DIR)
            .to(torch.bfloat16)
            .to(device)
        )
        # self.eos = [self.tokenizer.encode(s)[0] for s in EOF_STRINGS]
        self.eos = EOF_STRINGS
        self.max_length = max_length
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"
        self.skip_special_tokens = False

    def num_tokens(self, prompt):
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        return len(input_tokens[0])

    def generate(self, prompt, batch_size=10, temperature=1.0):
        input_str = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input_str, return_tensors="pt").to(
            self.device
        )

        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        raw_outputs = self.model.generate(
            input_tokens,
            max_length=self.max_length,
            do_sample=True,
            top_p=1.0,
            temperature=max(temperature, 1e-2),
            num_return_sequences=batch_size,
            stopping_criteria=scores,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


def scan_prompt(prompt_dirs: Path, existing_prompts: set, target: str = None):
    new_prompts = set()
    for target_dir in prompt_dirs.iterdir():
        if not target_dir.is_dir():
            continue
        if target is not None:
            if target_dir.name != target:
                continue
        for prompt_dir in target_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            for prompt_file in prompt_dir.iterdir():
                if not prompt_file.is_file():
                    continue
                if prompt_file.suffix != ".txt":
                    continue
                if prompt_file in existing_prompts:
                    continue
                new_prompts.add(prompt_file)
    return new_prompts


def clean_code(msg: str) -> str:
    if "```" not in msg:
        # the whole response message is a python program
        return msg
    codes = msg.split("```")
    for code in codes:
        # remove ```python
        code = code.split("\n", 1)[-1].strip()
        if len(code) > 0:
            return code
    return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompt/pytorch/step0")
    parser.add_argument("--output-dir", type=str, default="chatgpt/zero-shot")
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
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Set a specific target, default to all if unspecified.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--min_prompt_id", type=int, default=0)
    parser.add_argument("--max_prompt_id", type=int, default=10)

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Hard limit - max tokens for starcoder",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for starcoder"
    )

    args = parser.parse_args()
    arg_dict = vars(args)

    if args.hf_cache is not None:
        HF_CACHE_DIR = args.hf_cache
    else:
        HF_CACHE_DIR = os.environ.get("HF_HOME", "~/.cache/huggingface")

    # Set up output directories.
    prompt_dir = Path(args.prompt_dir)
    outdir = Path(args.output_dir)
    prompt_dir.mkdir(parents=True, mode=0o777, exist_ok=True)
    os.chmod(prompt_dir, 0o777)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o777)

    top_p = 1.0
    temperature = args.temperature
    device = args.device

    num = args.num
    max_tokens = args.max_tokens
    batch_size = args.batch_size
    sleep_time = 30

    _Model = StarCoder(device=device, max_length=max_tokens)
    logger = Logger(prompt_dir / "log.txt", is_print=True)

    logger.log("Arguments for starcoder service")
    for k, v in arg_dict.items():
        logger.log(f"  {k}: {v}")

    if args.target is not None:
        logger.log(f"Targeting: {args.target}")

    existing_prompts = set()
    while True:
        new_prompts = scan_prompt(prompt_dir, existing_prompts, args.target)
        if len(new_prompts) == 0:
            logger.log(f"No new prompts, sleep {sleep_time}s...")
            time.sleep(sleep_time)
            continue

        length = len(new_prompts)
        logger.log(f"Found {length} new prompts, start generating...")
        for idx, prompt_file in enumerate(new_prompts):
            existing_prompts.add(prompt_file)
            # Target name
            target_name = prompt_file.parent.parent.stem
            # Dir name
            dir_name = prompt_file.parent.stem
            # Opt name
            opt = prompt_file.stem
            # Skip if already exists
            gen_dir = outdir / target_name / dir_name / opt

            # If there is already a generated file, skip.
            if (gen_dir / f"{opt}_{batch_size}.py").exists():
                logger.log(
                    f"[{idx+1}/{length}] {target_name} - {opt}: skipped because its output already exists."
                )
                continue
            logger.log(f"[{idx+1}/{length}] {target_name} - {opt}: generating")

            code_idx = 0
            ret = {"opt": opt}
            ret["response"] = {}
            os.makedirs(gen_dir, exist_ok=True)

            i = 0
            cur_num = num
            n_batch_size = batch_size
            div = num / n_batch_size
            user_input = prompt_file.read_text(encoding="ascii", errors="ignore")
            (gen_dir / "prompt.txt").write_text(user_input)

            try:
                logger.log(f"Prompt tokens for {opt}: {_Model.num_tokens(user_input)}")
            except Exception as e:
                e = str(e)
                logger.log(f"[Error] Prompt tokens for {opt} is too long!!!")
                continue

            while cur_num > 0:
                fail = False
                while True:
                    # update batch size
                    n_batch_size = min(n_batch_size, cur_num)
                    logger.log(f"[{idx+1}/{length}] {opt} batch size: {n_batch_size}")
                    try:
                        t_start = time.time()
                        response = _Model.generate(
                            user_input, temperature=temperature, batch_size=n_batch_size
                        )
                        g_time = time.time() - t_start
                        logger.log(f"[{idx+1}/{length}] {opt} used time: {g_time}")
                        break
                    except Exception as e:
                        e = str(e)
                        if "CUDA out of memory" in str(e):
                            logger.log("    CUDA out of memory, reduce batch size.")
                        else:
                            logger.log(e)

                        # If batch size is 1, skip.
                        if n_batch_size == 1:
                            logger.log("Batch size is 1, skip!!!")
                            cur_num = 0
                            fail = True
                            break

                        div += 1
                        n_batch_size = int(cur_num / div)
                        time.sleep(2)
                if fail:
                    # If fail, skip this generation.
                    break

                cur_num -= n_batch_size
                div = max(div - 1, 1)

                msgs = response
                codes = []
                for msg in msgs:
                    code = clean_code(msg)
                    codes.append(code)
                    code_idx += 1
                    try:
                        (gen_dir / f"{opt}_{code_idx}.py").write_text(
                            code.encode("ascii", "ignore").decode()
                        )
                    except Exception:
                        pass
                (gen_dir / f"time.txt").write_text(str(g_time))
                ret["response"][i] = {"raw": response, "code": codes, "g_time": g_time}
                i += 1
            with open(gen_dir.parent / "outputs.json", "a") as f:
                f.write(json.dumps(ret, indent=4) + "\n")
