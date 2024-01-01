"""
The service backend for starcoder.

Please use the following command to start the service:
```
python executor_trigger.py | tee executor_trigger.log 2>&1
```
 
--json JSON file
--gen-dir Prompt DIR
--outdir Trigger result DIR
--llvm LLVM Compiled BIN Folder

The prompts of the gens are in `/JawTitan/whitefox-data/starcode/{target_name}/{step_name}/{gen_name}/{generated}`. 
The trigger executor result of the prompt runned will be in "/JawTitan/whitefox-data/starcoder/llvm-exe-result/trigger_test"
"""

import os
import re
import time
import json
from pathlib import Path

from executor import trigger_executor
from tqdm import tqdm  # Import tqdm for progress bar


def scan_gen(gen_dirs, existing_gens: set):
    new_gens = set()
    for target_dir in gen_dirs.iterdir():
        if not target_dir.is_dir():
            continue
        for gen_dir in target_dir.iterdir():
            if not gen_dir.is_dir():
                continue
            for gen_file in gen_dir.iterdir():
                if not gen_file.is_file():
                    continue
                # if gen_file.suffix != ".py" and gen_file.suffix != ".ll": continue
                if gen_file.suffix != ".ll":
                    continue
                if gen_file in existing_gens:
                    continue
                new_gens.add(gen_file)
    return new_gens


def info_dump(Statistics: dict, path=None):
    for k, v in Statistics.items():
        if isinstance(v, list):
            print(k, ":", len(v))
    target_lines_triggered = Statistics["target_lines_triggered"]
    # print("target_lines:", len(target_lines))
    # print("target_lines_triggered:", len(target_lines_triggered.keys()))
    print(
        "target_lines_triggered:",
        len([k for k, v in target_lines_triggered.items() if v != []]),
    )

    if path is not None:
        with open(path, "w") as f:
            f.write(json.dumps(Statistics, indent=4) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="example.json")
    parser.add_argument(
        "--gen-dir",
        type=str,
        default="/JawTitan/whitefox-data/starcoder-1000/llvm-opt/",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/JawTitan/whitefox-data/starcoder-1000/llvm-exe-result/trigger_test",
    )
    parser.add_argument(
        "--llvm", type=str, default=""
    )
    parser.add_argument(
        "--black_folder", type=str, default="chatgpt0test_hand_written_prompt"
    )  # deprecated
    parser.add_argument(
        "--target_folder", type=str, default="no_source_prompt0no_source_promptIDX"
    )
    parser.add_argument("--statistics-data", type=str, default="statistics.json")
    parser.add_argument("-j", type=int, default=8)

    # clang
    # ast->llvm ir->llvm ir->.s/.o->exe
    #  ->opt->llc->linker

    """
python executor_trigger.py --target_folder llvm-opt-run-1000 --statistics-data llvm-1000.json | tee executor_trigger.log 2>&1

python executor_trigger.py --target_folder no_source_prompt0no_source_promptIDX --statistics-data no_source_prompt.json | tee executor_trigger.log 2>&1
python executor_trigger.py --target_folder no_source_with_specific_ir_prompt0no_source_with_specific_ir_promptIDX --statistics-data no_source_with_specific_ir_prompt.json | tee executor_trigger_no_source_with_specific_ir_prompt.log 2>&1
python executor_trigger.py --target_folder no_source_with_eof_prompt0no_source_with_eof_promptIDX --statistics-data no_source_with_eof_prompt.json | tee executor_trigger.log 2>&1

python executor_trigger.py --target_folder mixed_prompt_for_starcoder0mixed_prompt_for_starcoderIDX --statistics-data mixed_prompt_for_starcoder.json | tee executor_trigger.log 2>&1

# for cpp
python executor_trigger.py --target_folder starcoder_cpp_deadarg0starcoder_cpp_deadargIDX --statistics-data starcoder_cpp_deadarg.json | tee starcoder_cpp_deadarg.log 2>&1

python executor_trigger.py --target_folder llvm-opt-runIDX0starcoder_cpp_deadargIDX --statistics-data starcoder_cpp_deadarg_new.json | tee executor_trigger.log 2>&1

python executor_trigger.py --target_folder llvm-opt-runIDX --statistics-data llvm-opt-run.json | tee llvm-opt-run.log 2>&1

python executor_trigger.py --target_folder llvm-opt-runIDX --statistics-data llvm-opt-run.json | tee llvm-opt-run_2023_09_27.log 2>&1

# for yarpgen
python executor_trigger.py --target_folder yarpgen_run_24 --statistics-data yarpgen_run_24.json | tee executor_trigger_yarpgen_run_24.log 2>&1

python executor_trigger.py --black_folder chatgpt0test_hand_written_prompt0no_source_with_eof_IDX --statistics-data no_source_prompt.json | tee executor_trigger.log 2>&1
python executor_trigger.py --black_folder chatgpt0test_hand_written_prompt0no_source_with_specific_ir_promptIDX0no_source_with_eofIDX --statistics-data no_source_prompt.json | tee executor_trigger.log 2>&1
    """

    args = parser.parse_args()

    llm_gen_path = args.gen_dir
    json_file = args.json
    llvm_source = args.llvm
    process = args.j

    # yarpgen cudasmith

    black_folder = list(args.black_folder.split("?"))
    for item in black_folder:
        if "IDX" in item:
            black_base = item[:-3]
            for i in range(1, 50):
                black_folder.append(black_base + str(i))

    target_folder = list(args.target_folder.split("?"))
    for item in target_folder:
        if "IDX" in item:
            target_base = item[:-3]
            for i in range(1, 50):
                target_folder.append(target_base + str(i))

    # special designed for 500 cpp, because we use '0' as the split symbol :(
    if "cpp" in args.target_folder:
        target_base = "starcoder_cpp_deadarg-500-"
        for i in range(1, 50):
            target_folder.append(target_base + str(i))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    os.chmod(outdir, 0o777)

    print(target_folder)

    gen_dir = Path(args.gen_dir)

    sleep_time = 15

    with open(json_file, "r") as file:
        data = json.load(file)

    json_statistics = os.path.join(outdir, args.statistics_data)
    if os.path.exists(json_statistics):
        with open(json_statistics, "r") as f:
            Statistics = json.load(f)
    else:
        Statistics = {
            "all_files": [],
            "grammatically correct": [],
            "grammatically uncorrect": [],
            "target_lines_triggered": {},
        }

    target_lines = []

    # target_lines_triggered = {}
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            # all the target lines
            target_lines.append(pass_info["target_line"])

            if not os.path.exists(json_statistics):
                # the target_line nums be triggered
                Statistics[pass_info["target_line"]] = 0
                Statistics["target_lines_triggered"][pass_info["target_line"]] = []
                # the map from the name to the target line
                Statistics[passname + "_oneshot" + f"_{index}"] = pass_info[
                    "target_line"
                ]

    existing_gen_files = set()
    while True:
        new_gen_files = scan_gen(gen_dir, existing_gen_files)
        # if (new_gen_files == set()):
        # new_gen_files = scan_gen(gen_dir, existing_gen_files)

        if len(new_gen_files) == 0:
            print(f"No new gen file, sleep {sleep_time}s...")
            info_dump(Statistics, json_statistics)
            time.sleep(sleep_time)
            continue

        length = len(new_gen_files)
        print(f"Found {length} new gen_files, start triggering...")
        # for idx, gen_file in enumerate(new_gen_files):
        for idx, gen_file in tqdm(
            enumerate(new_gen_files), total=length, desc="Processing gen_files"
        ):
            existing_gen_files.add(gen_file)
            # Target name
            target_name = gen_file.parent.parent.stem
            if target_name in black_folder:
                continue
            if target_name not in target_folder:
                continue
            # Dir name
            dir_name = gen_file.parent.stem
            # Opt name
            pattern = r"(.+)_oneshot_\d+"
            match = re.match(pattern, gen_file.stem)
            opt = match.group(0) if match is not None else dir_name

            try:  # if not yarpgen
                target_line = Statistics[opt]

                # out_pass_dir = outdir / target_name / opt
                out_pass_dir = os.path.join(outdir, target_name, opt)
                os.makedirs(out_pass_dir, exist_ok=True)
            except:
                target_line = "randomsk;djfk;ajfkdsjnfk;wjnfljsidfjl;kjfk;wjfk;ejofjlfjskjfklsjfisjilfjlikkkk"
                # out_pass_dir = outdir / target_name / opt
                out_pass_dir = os.path.join(outdir, target_name, opt)
                os.makedirs(out_pass_dir, exist_ok=True)

            # if os.path.exists(os.path.join(out_pass_dir, "success_" + gen_file.name + ".log")) or os.path.exists(os.path.join(out_pass_dir, "error_" + gen_file.name + ".log")):
            # continue

            # out_pass_file = out_pass_dir / gen_file.name
            trigger_executor(
                (str(out_pass_dir), str(gen_file.name)),
                str(gen_file),
                llvm_source,
                "opt",
                target_line,
                target_lines,
                Statistics,
            )

            info_dump(Statistics, json_statistics)

        # Close the tqdm progress bar
        # tqdm.close()
