import os
import re
import time
import json
from pathlib import Path

from executor import test_executor


# def scan_gen(json_folder, json_files, existing_files):
#     new_gen_files = set()

#     for json_file in json_files:
#         json_file = os.path.join(json_folder, json_file)
#         # print(json_file)
#         if os.path.exists(json_file):
#             with open(json_file, "r") as file:
#                 data = json.load(file)
#                 for k, v in data["target_lines_triggered"].items():
#                     if isinstance(v, list):
#                         for item in v:
#                             if item not in existing_files:
#                                 new_gen_files.add(item)
#         # print("len: ", len(new_gen_files))
#         # print("existing file: ", len(existing_files))

#     return new_gen_files


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
    print(f"{len(Statistics.keys())} has been tested and saved")

    if path is not None:
        with open(path, "a") as f:
            f.write(json.dumps(Statistics) + "\n")


def info_read(path=None):
    alldata = {}
    if path is not None:
        with open(path, "r") as f:
            lines = f.readlines()
            for l in lines:
                data = json.loads(l)
                alldata.update(data)
    return alldata


"""
python --json_folder
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-folder",
        type=str,
        default="/JawTitan/whitefox-data/starcoder-1000/llvm-exe-result/trigger_test",
    )
    parser.add_argument(
        "--jsons",
        type=str,
        default="no_source_with_specific_ir_prompt.json_AND_no_source_prompt.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/JawTitan/whitefox-data/llvm-exe-result/diff_test1",
    )
    parser.add_argument(
        "--gen-dir",
        type=str,
        default="/JawTitan/whitefox-data/starcoder-1000/llvm-opt/",
    )
    args = parser.parse_args()

    # python executor_test.py --jsons mixed_prompt_for_starcoder.json --out_dir  /JawTitan/whitefox-data/starcoder/llvm-exe-result/diff_test_mixed_nl
    # python executor_test.py --jsons starcoder_cpp_deadarg.json --out_dir  /JawTitan/whitefox-data/starcoder/llvm-exe-result/diff_test_cpp_code
    # python executor_test.py --jsons llvm-opt-run_2023_09_27.json --out_dir  /JawTitan/whitefox-data/starcoder/llvm-exe-result/diff_test_llvm_opt_run
    # python executor_test.py --jsons llvm-1000.json --out_dir  /JawTitan/whitefox-data/starcoder-1000/llvm-exe-result/diff_test_llvm_opt_run

    sleep_time = 30

    json_files = list(args.jsons.split("_AND_"))
    print(json_files)
    json_folder = args.json_folder
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)

    out_json_path = os.path.join(out_dir, "diff_result.jsonl")
    if os.path.exists(out_json_path):
        print("find json: ", out_json_path)
        out_json_dict = info_read(out_json_path)
    else:
        print("don't find json, build one: ", out_json_path)
        out_json_dict = {}

    all_trigger_files = []
    existing_files = set(out_json_dict.keys())
    gen_dirs = Path(args.gen_dir)

    while True:
        new_gen_files = scan_gen(gen_dirs, existing_files)

        if len(new_gen_files) == 0:
            print(f"No new gen file, sleep {sleep_time}s...")
            time.sleep(sleep_time)
            continue

        length = len(new_gen_files)
        print(f"Found {length} new gen_files, start testing...")
        for idx, gen_file in enumerate(new_gen_files):
            existing_files.add(gen_file)

            # llc -> compare -> exe -> compare
            log = test_executor(gen_file, str(out_dir), timeout=30)

            if log == None:
                continue

            info_dump(log, out_json_path)
            out_json_dict.update(log)
