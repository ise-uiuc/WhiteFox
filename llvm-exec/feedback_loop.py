from pathlib import Path
import subprocess as sp
import argparse
import os
import json
import time
from executor import trigger_executor
import re
from gen_prompt import gen_prompt_nl2test
import numpy as np
import random

BATCH_SIZE = 10

def scan_gen(gen_dirs, existing_gens: set):
    """Get the new generated files with suffix `.c`."""  
    new_gens = set()
    if not gen_dirs.exists():
        return new_gens
    for target_dir in gen_dirs.iterdir():
        if not target_dir.is_dir():
            continue
        for gen_file in target_dir.iterdir():
            if not gen_file.is_file():
                continue
            if gen_file.suffix != ".c":  
                continue
            if gen_file in existing_gens:
                continue
            new_gens.add(gen_file)
    return new_gens

def gen_prompt(data: dict, gen_dir_path=None, example_dict=None):
    print(example_dict)
    for passname, pass_infos in data.items():
        for index, _ in enumerate(pass_infos["hints"]):
            passname_in_pool = f"{passname}_oneshot_{index}"
            if passname_in_pool in example_dict:
                examples = example_dict[passname_in_pool]
                gen_prompt_nl2test(
                    passname, index, "ollama_c_feedback", gen_dir_path, examples
                )
            else:
                gen_prompt_nl2test(
                    passname,
                    index,
                    "ollama_c_deadarg",
                    gen_dir_path,
                    None,
                )

def update_example_pool(example_pool: dict, stat: dict, chosen: dict):
    print(example_pool)
    for passname in example_pool.keys():
        pass_trigger_line = stat[passname]
        trigger_files = stat["target_lines_triggered"][pass_trigger_line]
        if passname not in chosen or len(chosen[passname]) == 0:
            assert len(example_pool[passname]) == 0, "This should not happen"
            example_pool[passname] = {}
            # This is newly triggered.
            for trigger_file in trigger_files:
                example_pool[passname][trigger_file] = {"alpha": 1, "beta": 1}
        else:
            filter_trigger_files = []
            # Filter trigger files.
            for trigger_file in trigger_files:
                print(passname, trigger_file)
                if passname in str(trigger_file):
                    filter_trigger_files.append(trigger_file)
                else:
                    example_pool[passname][trigger_file] = {"alpha": 1, "beta": 1}

            if len(filter_trigger_files) > BATCH_SIZE:
                raise ValueError(
                    f"Trigger files {len(filter_trigger_files)} > {BATCH_SIZE}"
                )

            # Update the example pool.
            sum_alpha, sum_beta = 0, 0
            for chosen_file in chosen[passname]:
                example_pool[passname][chosen_file]["alpha"] += len(
                    filter_trigger_files
                )
                example_pool[passname][chosen_file]["beta"] += BATCH_SIZE - len(
                    filter_trigger_files
                )
                sum_alpha += example_pool[passname][chosen_file]["alpha"]
                sum_beta += example_pool[passname][chosen_file]["beta"]

            avg_alpha = sum_alpha / len(chosen[passname])
            avg_beta = sum_beta / len(chosen[passname])
            for new_file in filter_trigger_files:
                example_pool[passname][new_file] = {
                    "alpha": avg_alpha,
                    "beta": avg_beta,
                }
    print(example_pool)

def select_examples(examples, num, use_rl):
    length = len(examples)
    keys = list(examples.keys())
    if num > length:
        idxs = list(range(length))
    elif not use_rl:
        idxs = random.sample(range(length), num)
    else:
        beta_list = []
        for k in keys:
            alpha = examples[k]["alpha"] if "alpha" in examples[k] else 1
            beta = examples[k]["beta"] if "beta" in examples[k] else 1
            beta_list.append(np.random.beta(alpha, beta))
        idxs = np.argsort(beta_list)[-num:].tolist()
    return [keys[idx] for idx in idxs]

def sample_example_pool(example_pool: dict, shot):
    chosen = {}
    examples_dict = {}
    for passname in example_pool:
        if len(example_pool[passname]) == 0:
            continue
        chosen[passname] = select_examples(example_pool[passname], shot, True)
        examples_dict[passname] = []
        for chosen_file in chosen[passname]:
            c_file = Path(str(chosen_file).replace(".ll", ".c"))  
            with open(c_file, "r") as file:
                examples_dict[passname].append(file.read())
    return chosen, examples_dict

def translate2ir(gen_file: Path):
    c_gen_file = gen_file.parent / (gen_file.stem + ".c")  
    ll_gen_file = gen_file.parent / (gen_file.stem + ".ll")
    if not c_gen_file.exists():
        c_gen_file.write_text(gen_file.read_text())
    command = [
        "clang",  # Changed from clang++ to clang
        "-O0",
        "-mllvm",
        "--debug",
        str(c_gen_file),
        "-S",
        "-emit-llvm",
        "-o",
        str(ll_gen_file),
    ]
    try:
        result = sp.run(command, capture_output=True, timeout=60)
    except Exception as e:
        print(f"Exception for translating {gen_file}:\n  {e}")
        return False
    return result.returncode == 0

def execute_trigger(step_gen_dir: Path, number: int, outdir: Path):
    # Initialize the statistics.
    existing_gen_files = set()
    Statistics = {
        "all_files": [],
        "grammatically correct": [],
        "grammatically uncorrect": [],
        "target_lines_triggered": {},
    }
    target_lines = []
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            target_lines.append(pass_info["target_line"])
            Statistics[pass_info["target_line"]] = 0
            Statistics["target_lines_triggered"][pass_info["target_line"]] = []
            Statistics[passname + "_oneshot" + f"_{index}"] = pass_info["target_line"]

    while len(existing_gen_files) < number:
        new_gen_files = scan_gen(step_gen_dir, existing_gen_files)
        if len(new_gen_files) == 0:
            print("No new gen files found, sleep for 30 seconds")
            time.sleep(30)
            continue
        existing_gen_files.update(new_gen_files)
        for gen_file in new_gen_files:
            print(gen_file)
            if not translate2ir(gen_file):
                continue
            ll_gen_file = gen_file.parent / (gen_file.stem + ".ll")
            dir_name = gen_file.parent.stem
            pattern = r"(.+)_oneshot_\d+"
            match = re.match(pattern, gen_file.stem)
            opt = match.group(0) if match is not None else dir_name

            target_line = Statistics[opt]
            target_name = gen_file.parent.parent.stem
            out_pass_dir = os.path.join(outdir, target_name, opt)
            os.makedirs(out_pass_dir, exist_ok=True)

            trigger_executor(
                (str(out_pass_dir), str(ll_gen_file.name)),
                str(ll_gen_file),
                llvm_source / "build" / "bin",
                "opt",
                target_line,
                target_lines,
                Statistics,
            )
    return Statistics

def feedback_step(
    data: dict, root_prompt_dir: Path, root_gen_dir: Path, step: int, example_pool: dict
):
    gen_dir = root_gen_dir / f"step{step}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    trigger_dir = root_gen_dir / f"step{step}_trigger"
    trigger_dir.mkdir(parents=True, exist_ok=True)

    chosen, examples_dict = sample_example_pool(example_pool, 3)
    gen_prompt(data, root_prompt_dir / f"step{step}", examples_dict)

    stat = execute_trigger(gen_dir, BATCH_SIZE * len(data.keys()), trigger_dir)
    with open(trigger_dir / f"statistics-step{step}.json", "w") as file:
        json.dump(stat, file, indent=4)
    update_example_pool(example_pool, stat, chosen)
    with open(trigger_dir / f"example_pool-step{step}.json", "w") as file:
        json.dump(example_pool, file, indent=4)
    return stat