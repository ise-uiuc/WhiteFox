import os
import re
import time
import json
from pathlib import Path
from executor import trigger_executor
from tqdm import tqdm

def scan_gen(gen_dirs, existing_gens: set):
    """Get the new generated files with suffix `.ll`."""
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
    print(
        "target_lines_triggered:",
        len([k for k, v in target_lines_triggered.items() if v != []])
    )

    if path is not None:
        with open(path, "w") as f:
            f.write(json.dumps(Statistics, indent=4) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="example.json")
    parser.add_argument("--gen-dir", type=str, default="prompt/demo/llvm-opt/")
    parser.add_argument("--outdir", type=str, default="ollama/output/trigger_test")
    parser.add_argument("--llvm", type=str, default="")
    parser.add_argument("--black_folder", type=str, default="")
    parser.add_argument("--target_folder", type=str, default="ollama_c_deadarg0ollama_c_deadargIDX")
    parser.add_argument("--statistics-data", type=str, default="statistics.json")
    parser.add_argument("-j", type=int, default=8)

    args = parser.parse_args()

    llm_gen_path = args.gen_dir
    json_file = args.json
    llvm_source = args.llvm
    process = args.j

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
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            target_lines.append(pass_info["target_line"])

            if not os.path.exists(json_statistics):
                Statistics[pass_info["target_line"]] = 0
                Statistics["target_lines_triggered"][pass_info["target_line"]] = []
                Statistics[passname + "_oneshot" + f"_{index}"] = pass_info["target_line"]

    existing_gen_files = set()
    while True:
        new_gen_files = scan_gen(gen_dir, existing_gen_files)

        if len(new_gen_files) == 0:
            print(f"No new gen file, sleep {sleep_time}s...")
            info_dump(Statistics, json_statistics)
            time.sleep(sleep_time)
            continue

        length = len(new_gen_files)
        print(f"Found {length} new gen_files, start triggering...")
        
        for idx, gen_file in tqdm(
            enumerate(new_gen_files), total=length, desc="Processing gen_files"
        ):
            existing_gen_files.add(gen_file)
            target_name = gen_file.parent.parent.stem
            if target_name in black_folder:
                continue
            if target_name not in target_folder:
                continue
                
            dir_name = gen_file.parent.stem
            pattern = r"(.+)_oneshot_\d+"
            match = re.match(pattern, gen_file.stem)
            opt = match.group(0) if match is not None else dir_name

            try:
                target_line = Statistics[opt]
                out_pass_dir = os.path.join(outdir, target_name, opt)
                os.makedirs(out_pass_dir, exist_ok=True)
            except:
                target_line = "randomsk;djfk;ajfkdsjnfk;wjnfljsidfjl;kjfk;wjfk;ejofjlfjskjfklsjfisjilfjlikkkk"
                out_pass_dir = os.path.join(outdir, target_name, opt)
                os.makedirs(out_pass_dir, exist_ok=True)

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