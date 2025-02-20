import subprocess
import os
import json
import glob
import time
from utils import find_passes_from_ll_file, update_ll_file, extract_code_from_markdown

def trigger_executor(
    out_pass_folder_file_tuple: tuple,
    filename: str,
    llvm_source: str,
    binary: str,
    target_line=None,
    target_lines=[],
    Statistics={},
):
    assert binary in [
        "llc",
        "clang",
        "opt",
    ], "binary must be the opt, llc or clang now!"

    run_file = os.path.join(llvm_source, binary)
    assert os.path.exists(run_file), f"file {run_file} not exist!"

    out_dir, out_name = out_pass_folder_file_tuple
    err_path = os.path.join(out_dir, "error_" + out_name + ".log")
    succ_path = os.path.join(out_dir, "success_" + out_name + ".log")

    def run_command(filename: str, run_file: str):
        if filename[-3:] == ".py":
            if not os.path.exists(str(filename) + ".ll"):
                with open(filename, "r") as f:
                    text = f.read()
                    code = extract_code_from_markdown(
                        text.encode("ascii", "ignore").decode(), "c"  # Changed from llvm to c
                    )
                    out_dir, out_name = out_pass_folder_file_tuple
                    with open(os.path.join(out_dir, out_name + ".c"), "w") as wf:  # Changed to .c
                        wf.write(code)
                    with open(os.path.join(out_dir, out_name + ".ll"), "w") as wf:
                        # Compile C to LLVM IR
                        compile_result = subprocess.run(
                            ["clang", "-O0", "-S", "-emit-llvm", os.path.join(out_dir, out_name + ".c"), "-o", "-"],
                            capture_output=True,
                            text=True
                        )
                        if compile_result.returncode == 0:
                            wf.write(compile_result.stdout)
                        else:
                            print(f"Failed to compile C to LLVM IR: {compile_result.stderr}")
                            return
                    filename = os.path.join(out_dir, out_name + ".ll")
            else:
                return

        if binary == "llc":
            command = [
                f"{run_file}",
                filename,
                "-O3",
                "--debug",
                "-mtriple=x86_64-unknown-linux-gnu",
            ]
        elif binary == "clang":  # clang
            out_file = os.path.join(
                "source-code-data/llvm/llvm-func-body", os.path.basename(filename)
            )
            if update_ll_file(filename, out_file):
                filename = out_file
            command = [
                f"{run_file}",
                "--debug",
                filename,
            ]
        else:  # opt
            command = [
                f"{run_file}",
                "--debug",
                "-O3",
                filename,
            ]

        try:
            result = subprocess.run(command, capture_output=True, timeout=100)
        except:
            print("time out")
            return

        Statistics["all_files"].append(filename)

        if result.returncode == 0:
            string_write_to_file = ""
            out_dir, out_name = out_pass_folder_file_tuple
            print("<--------------------------------------------->")
            print(f"{binary} {os.path.basename(filename)} Command executed successfully.")
            string_write_to_file += (
                f"{binary} {os.path.basename(filename)} Command executed successfully.\n"
            )
            Statistics["grammatically correct"].append(filename)
            if binary == "opt":
                for target_line_ in target_lines:
                    if target_line_ in result.stderr.decode():
                        if target_line_ != target_line:
                            Statistics["target_lines_triggered"][target_line_].append(
                                filename
                            )
                            print(
                                f"target line {target_line_} in file {filename} is triggered!"
                            )
                            string_write_to_file += f"target line {target_line_} in file {filename} is triggered!\n"
                            Statistics[target_line_] += 1
                if target_line in result.stderr.decode():
                    print(
                        f"target_line we want {target_line} in file {filename} is triggered!"
                    )
                    Statistics["target_lines_triggered"][target_line].append(filename)
                    string_write_to_file += f"target_line we want {target_line} in file {filename} is triggered!\n"
                    Statistics[target_line] += 1
            with open(os.path.join(out_dir, "success_" + out_name + ".log"), "w") as f:
                f.write(string_write_to_file)
        else:
            string_write_to_file = ""
            out_dir, out_name = out_pass_folder_file_tuple
            Statistics["grammatically uncorrect"].append(filename)
            print("<--------------------------------------------->")
            print(
                f"{binary} {os.path.basename(filename)} Command encountered an error with exit status: {result.returncode}"
            )
            string_write_to_file += f"{binary} {os.path.basename(filename)} Command encountered an error with exit status: {result.returncode}\n"
            str_command = " ".join(str(item) for item in command)
            print(f"the command is {str_command}")
            string_write_to_file += f"the command is {str_command}\n"
            print(f"the error file is: {os.path.basename(filename)}")
            string_write_to_file += f"the error file is: {os.path.basename(filename)}\n"
            print("Error output:")
            print(result.stderr.decode())
            string_write_to_file += "Error output:\n"
            string_write_to_file += f"{result.stderr.decode()}\n"
            with open(os.path.join(out_dir, "error_" + out_name + ".log"), "w") as f:
                f.write(string_write_to_file)

    run_command(filename, run_file)