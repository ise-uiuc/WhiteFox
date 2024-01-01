"""
add instrument to the source code of llvm
"""
import json
import argparse
import os

from utils import count_prefix_indent


def add_instrument(data: dict):
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            source_code_file = pass_info["codes"][0]

            add_line_after_code(source_code_file, pass_info["target_line"])


def add_line_after_code(file_path, target_code):
    # Read the contents of the C++ file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the location of the target code
    target_line = None
    for i, line in enumerate(lines):
        if target_code in line:
            target_line = i
            break

    if target_line is None:
        print(f"Target code not found in the file{file_path}")
        return

    if target_code in lines[target_line + 1]:
        return None

    lines.insert(
        target_line + 1,
        count_prefix_indent(lines[target_line]) * " "
        + f'LLVM_DEBUG(dbgs() << "{target_code}'
        + "\\"
        + "n"
        + '"'
        + ");\n",
    )

    # Write the modified contents back to the file
    with open(file_path, "w") as file:
        file.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="example.json")
    args = parser.parse_args()
    json_file = args.json
    with open(json_file, "r") as file:
        data = json.load(file)

    # add_instrument
    target_dict = add_instrument(data)
    # print(target_dict)
    # add_line_after_code("/home/lry/projects/JIT-parser/source-code-data/llvm/llvm-func-body/DeadArgumentEliminationPass0.cpp", "Arg.replaceAllUsesWith(PoisonValue::get(Arg.getType()));")
