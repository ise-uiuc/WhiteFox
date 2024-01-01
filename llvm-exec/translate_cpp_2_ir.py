import os
import re
import time
import json
from pathlib import Path
import subprocess

from executor import trigger_executor
from utils import extract_code_from_markdown

def scan_gen(gen_dirs, target_folder, existing_gens: set):
    new_gens = set()
    for target_dir in gen_dirs.iterdir():
        if not target_dir.is_dir(): 
            continue
        if str(target_dir.stem) not in target_folder:
            continue
        for gen_dir in target_dir.iterdir():
            if not gen_dir.is_dir(): continue
            for gen_file in gen_dir.iterdir():
                if not gen_file.is_file(): continue
                if gen_file.suffix != ".py": continue
                if gen_file in existing_gens: continue
                new_gens.add(gen_file)
    return new_gens

if __name__ == "__main__":
    '''
    python translate_cpp_2_ir.py
    python translate_cpp_2_ir.py --target-folder llvm-opt-run0llvm-opt-runIDX
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-folder", type=str, default="starcoder_cpp_deadarg0starcoder_cpp_deadargIDX")
    parser.add_argument("--gen-dir", type=str, default="/JawTitan/whitefox-data/starcoder/llvm-opt/")
    parser.add_argument("--do-extract", action="store_true")
    parser.add_argument("--stop-num", type=int, default=5000)
    args = parser.parse_args()

    target_folder = args.target_folder
    gen_dir = Path(args.gen_dir)
    sleep_time = 30

    target_folder = list(args.target_folder.split("?"))
    for item in target_folder:
        if 'IDX' in item:
            target_base = item[:-3]
            for i in range(1, 50):
                target_folder.append(target_base + str(i))

    existing_gen_files = set()
    right_file = 0
    total_file = 0

    while True:
        new_gen_files = scan_gen(gen_dir, target_folder, existing_gen_files)


        if len(new_gen_files) == 0:
            print(f"No new gen file, sleep {sleep_time}s...")
            print(f"files syntax right: {right_file}, total file: {total_file}")
            time.sleep(sleep_time)
            continue

        length = len(new_gen_files)
        total_file += length
        print(f"Found {length} new gen_files, start translating...")

        for idx, gen_file in enumerate(new_gen_files):
            existing_gen_files.add(gen_file)

            if os.path.exists(str(gen_file) + ".ll"):
                continue

            with open(gen_file, "r") as f:
                text = f.read()
                if args.do_extract:
                    code = extract_code_from_markdown(text, code="cpp")
                else:
                    code = text
            with open(str(gen_file) + ".cpp", "w") as f:
                f.write(code)

            command = ["clang++", "-O0", "-mllvm", "--debug", str(gen_file) + ".cpp", "-S", "-emit-llvm", "-o", str(gen_file) + ".ll"]
            print(f"[{idx}:{len(new_gen_files)}]")
            print(" ".join(command))
            result = subprocess.run(command, capture_output=True)
            print(result.stderr)

            if result.returncode == 0:
                right_file += 1

        if args.stop_num > 0 and total_file >= args.stop_num:
            break
