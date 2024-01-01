import subprocess
import os
import json
import glob
import time

from utils import find_passes_from_ll_file, update_ll_file, extract_code_from_markdown


def example():
    # Run a command and wait for it to complete
    result = subprocess.run(["ls", "-l"], capture_output=True, text=True)
    print(result.stdout)

    # Run a command and capture its output
    output = subprocess.check_output(["pwd"], text=True)
    print(output)

    # Run a command and capture its return code
    return_code = subprocess.call(["mkdir", "new_directory"])
    print(return_code)

    return_code = subprocess.call(["mkdir", "new_directory"])
    print(return_code)

    return_code = subprocess.call(["rm", "-rf", "new_directory"])
    print(return_code)


def check_official_pass(path: str, json_file: str, opt_file: str, outfile="test.log"):
    with open(outfile, "w") as test:
        pass

    with open(json_file, "r") as file:
        data = json.load(file)

    all_target_lines = []  # collect all target lines
    target_dict = dict()  # collect the target_line:example_ll files
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            all_target_lines.append(pass_info["target_line"])

            # if pass_info['func'][-1] in target_dict:
            #     target_dict[pass_info['func'][-1]].append(pass_info['target_line'])
            # else:
            #     target_dict[pass_info['func'][-1]] = [pass_info['target_line']]

    find_set = set()
    files = glob.glob(path + "/*.ll")

    for file in files:
        result = subprocess.run(
            [
                f"{opt_file}",
                file,
                f"-passes={find_passes_from_ll_file(file)}",
                "--debug",
                "-S",
                "-o",
                "out.ll",
            ],
            capture_output=True,
            text=True,
        )

        num = 0
        for target in all_target_lines:
            loc = result.stderr.find(target)
            if loc != -1:
                # print(target, loc)
                find_set.add(target)
                num += 1

        with open(outfile, "a") as log:
            if result.stderr is not None:
                log.write(result.stderr)

    # if len(find_set) == len(all_target_lines):
    # print("all here!")
    find_lines = []
    not_find_lines = []
    with open(outfile, "r") as file:
        out = file.read()
        for target_line in all_target_lines:
            if target_line in out:
                find_lines.append(target_line)
                # print(f"find target line: {target_line}")
            else:
                not_find_lines.append(target_line)
                # print(f"target line {target_line} not found")

    print("<--------------------------------------------->")
    print("so, those target lines are triggerd :) \n")
    print("\n".join(find_lines))
    print("<--------------------------------------------->")
    print("so, those target lines are not triggerd :(\n")
    print("\n".join(not_find_lines))
    print("<--------------------------------------------->")

    pass


from utils import count_characters


def trigger_line_check(
    run_file: str, file_path: str, target_lines: str, Statistics: dict
):
    """
    run_file: where the opt is, such as /home/lry/projects/llvm-project/build/bin/opt
    path
    return true if triggerd
    """

    assert os.path.exists(run_file), f"file {run_file} not exist!"

    file = file_path

    command = [f"{run_file}", "--debug", "-O3", file]

    try:
        result = subprocess.run(command, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(None, e.stderr, e.returncode)

    # Check the exit status and print the output
    if result.returncode == 0:
        print(
            f"{os.path.basename(run_file)} {os.path.basename(file)} Command executed successfully."
        )
        for target_line_ in target_lines:
            if target_line_ in result.stderr.decode():
                print(f"target line {target_line_} in file {file} is triggered!")
                if Statistics[target_line_] == "":
                    Statistics[target_line_] = file
                else:
                    ori_len = count_characters(Statistics[target_line_])
                    cur_len = count_characters(file)
                    if ori_len > cur_len:
                        Statistics[target_line_] = file
    pass


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
                        text.encode("ascii", "ignore").decode(), "llvm"
                    )
                    out_dir, out_name = out_pass_folder_file_tuple
                    with open(os.path.join(out_dir, out_name + ".ll"), "w") as wf:
                        wf.write(code)
                    filename = os.path.join(out_dir, out_name + ".ll")
                    assert 0, "please check the file"
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
                "--debug",  # f"-passes={find_passes_from_ll_file(example_file)}", \
                "-O3",
                filename,
            ]

        # result = subprocess.run(command, capture_output=True, text=True)
        # stdout = result.stdout.decode('utf-8', errors='ignore')  # Convert to utf-8 if needed
        # stderr = result.stderr.decode('utf-8', errors='ignore')  # Convert to utf-8 if needed
        try:
            result = subprocess.run(command, capture_output=True, timeout=100)
            # stdout = result.stdout  # Bytes output
            # stderr = result.stderr  # Bytes output
            # print(stdout.decode())
            # print(stderr.decode())
            # return stdout, stderr, result.returncode
        # except subprocess.CalledProcessError as e:
        except:
            # print(None, e.stderr, e.returncode)
            print("time out")
            return

        Statistics["all_files"].append(filename)

        # Check the exit status and print the output
        if result.returncode == 0:
            string_write_to_file = ""
            out_dir, out_name = out_pass_folder_file_tuple
            print("<--------------------------------------------->")
            print(f"{binary} {os.path.basename(filename)} Command executed successfully.")
            string_write_to_file += (
                f"{binary} {os.path.basename(filename)} Command executed successfully.\n"
            )
            # Statistics = {"all_files":[], "grammatically correct":[]}
            # Statistics["grammatically correct"].append(os.path.basename(file))
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
            # Statistics["grammatically uncorrect"].append(os.path.basename(file))
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


def run_command_with_timeout(command, timeout):
    try:
        process = subprocess.Popen(command, shell=True)

        start_time = time.time()
        while True:
            return_code = process.poll()
            if return_code is not None:
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                process.terminate()
                raise TimeoutError(f"timmout ({timeout}s)")

            time.sleep(1)

        return return_code

    except Exception as e:
        print(f"exception : {e}")
        return -1


def test_executor(
    file: str, out_dir="/JawTitan/whitefox-data/starcoder/llvm-exe-result/", timeout=300
):
    """
    llc compile error:
    exe error:
    """

    from pathlib import Path

    file = Path(file)
    out_folder = os.path.join(
        out_dir, "_".join([file.stem, file.parent.stem, file.parent.parent.stem])
    )
    # out_folder = os.path.join(out_dir, [file.stem,file.parent.stem,file.parent.parent.stem].join("_"))
    if os.path.exists(out_folder):
        return
    else:
        os.mkdir(out_folder)
    file_before_opt = os.path.join(out_folder, "file_before_opt.ll")
    file_after_opt = os.path.join(out_folder, "file_after_opt.ll")
    exe_before_opt = os.path.join(out_folder, "exe_before_opt")
    exe_after_opt = os.path.join(out_folder, "exe_after_opt")

    commands = {
        "cp_ori": ["cp", file, file_before_opt],
        "opt": [
            "opt",
            file,
            "-O3",
            "--debug",
            "-mtriple=x86_64-unknown-linux-gnu",
            "-o",
            file_after_opt,
        ],
        "llc_before": [
            "llc",
            file,
            "-O3",
            "-filetype=obj",
            "--debug",
            "-mtriple=x86_64-unknown-linux-gnu",
            "-o",
            exe_before_opt + ".o",
        ],
        "llc_after": [
            "llc",
            file_after_opt,
            "-O3",
            "-filetype=obj",
            "--debug",
            "-mtriple=x86_64-unknown-linux-gnu",
            "-o",
            exe_after_opt + ".o",
        ],
        "linker_before": ["clang", exe_before_opt + ".o", "-o", exe_before_opt],
        "linker_after": ["clang", exe_after_opt + ".o", "-o", exe_after_opt],
        "exe_before": [f"{exe_before_opt}"],
        "exe_after": [f"{exe_after_opt}"],
    }

    # check if the executor's result is right

    file = str(file)
    out_dict = {file: {}}

    for k, v in commands.items():
        # print(" ".join(v))
        print(" ".join([str(item) for item in v]))
        k = str(k)
        try:
            if "exe" in k and not os.path.exists(commands[k][0]):
                continue
            if "exe" in k:
                returncode = run_command_with_timeout(commands[k], timeout)
                print(returncode)
                out_dict[file][k] = {}
                out_dict[file][k]["returncode"] = returncode
            else:
                result = subprocess.run(commands[k], capture_output=True)
                print(result.returncode)
                out_dict[file][k] = {}
                out_dict[file][k]["returncode"] = result.returncode

        except subprocess.CalledProcessError as e:
            print(None, e.stderr, e.returncode)
            continue

    return out_dict
    # out_dict[file][k]["stdout"] = result.stdout.decode()
    # out_dict[file][k]["stderr"] = result.stderr.decode()


def draft_executor(
    path: str,
    llvm_source: str,
    binary: str,
    process_num=16,
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

    files = glob.glob(path + "/*.ll")

    def run_command(file: str, run_file: str):
        if binary == "llc":
            command = [
                f"{run_file}",
                file,
                "-O3",
                "--debug",
                "-mtriple=x86_64-unknown-linux-gnu",
            ]
        elif binary == "clang":  # clang
            out_file = os.path.join(
                "source-code-data/llvm/llvm-func-body", os.path.basename(file)
            )
            if update_ll_file(file, out_file):
                file = out_file
            command = [
                f"{run_file}",
                "--debug",
                file,
            ]
        else:  # opt
            command = [
                f"{run_file}",
                "--debug",  # f"-passes={find_passes_from_ll_file(example_file)}", \
                "-O3",
                file,
            ]

        # result = subprocess.run(command, capture_output=True, text=True)
        # stdout = result.stdout.decode('utf-8', errors='ignore')  # Convert to utf-8 if needed
        # stderr = result.stderr.decode('utf-8', errors='ignore')  # Convert to utf-8 if needed
        try:
            result = subprocess.run(command, capture_output=True)
            # stdout = result.stdout  # Bytes output
            # stderr = result.stderr  # Bytes output
            # print(stdout.decode())
            # print(stderr.decode())
            # return stdout, stderr, result.returncode
        except subprocess.CalledProcessError as e:
            print(None, e.stderr, e.returncode)
            # return None, e.stderr, e.returncode

        Statistics["all_files"].append(os.path.basename(file))

        # Check the exit status and print the output
        if result.returncode == 0:
            print("<--------------------------------------------->")
            print(f"{binary} {os.path.basename(file)} Command executed successfully.")
            # Statistics = {"all_files":[], "grammatically correct":[]}
            # Statistics["grammatically correct"].append(os.path.basename(file))
            Statistics["grammatically correct"].append(file)
            if binary == "opt":
                for target_line_ in target_lines:
                    if target_line_ in result.stderr.decode():
                        if target_line_ != target_line:
                            print(
                                f"target line {target_line_} in file {file} is triggered!"
                            )
                            Statistics[target_line_] += 1
                if target_line in result.stderr.decode():
                    print(
                        f"target_line we want {target_line} in file {file} is triggered!"
                    )
                    Statistics[target_line] += 1
        else:
            # Statistics["grammatically uncorrect"].append(os.path.basename(file))
            Statistics["grammatically uncorrect"].append(file)
            print("<--------------------------------------------->")
            print(
                f"{binary} {os.path.basename(file)} Command encountered an error with exit status: {result.returncode}"
            )
            str_command = " ".join(command)
            print(f"the command is {str_command}")
            print(f"the error file is: {os.path.basename(file)}")
            print("Error output:")
            print(result.stderr.decode())

    for file in files:
        run_command(file, run_file)

    # from multiprocessing import Process
    # # Number of processes to run concurrently (you can adjust this based on your system's capabilities)
    # num_processes = process_num

    # processes = []
    # for i in range(0, len(files), num_processes):
    #     batch_files = files[i:i + num_processes]
    #     for file in batch_files:
    #         process = Process(target=run_command, args=(file, run_file))
    #         processes.append(process)
    #         process.start()

    #     for process in processes:
    #         process.join()

    # subprocess.run("rm out", capture_output=True, text=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="example.json")
    parser.add_argument("--dir", type=str, default="chatgpt/one-shot")
    parser.add_argument(
        "--opt", type=str, default=""
    )
    parser.add_argument(
        "--clang", type=str, default=""
    )
    parser.add_argument(
        "--llvm", type=str, default=""
    )
    parser.add_argument("-j", type=int, default=8)

    args = parser.parse_args()

    json_file = args.json
    llm_generation_path = args.dir
    opt_file = args.opt
    clang_file = args.clang
    llvm_source = args.llvm
    process = args.j

    with open(json_file, "r") as file:
        data = json.load(file)

    Statistics = {
        "all_files": [],
        "grammatically correct": [],
        "grammatically uncorrect": [],
    }

    target_lines = []
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            target_lines.append(pass_info["target_line"])
            Statistics[pass_info["target_line"]] = 0

    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            single_opt_folder = os.path.join(
                llm_generation_path, passname + "_oneshot" + f"_{index}"
            )
            if not os.path.exists(single_opt_folder):
                continue
            print(f"{single_opt_folder} exist!")
            target_line = pass_info["target_line"]

            draft_executor(
                single_opt_folder,
                llvm_source,
                "opt",
                16,
                target_line,
                target_lines,
                Statistics,
            )

    target_lines_triggered = []
    for k, v in Statistics.items():
        if isinstance(v, list):
            print(k, ":", len(v))
        else:
            print(k, ":", v)
            if v != 0:
                target_lines_triggered.append(v)

    print("target_lines:", len(target_lines))
    print("target_lines_triggered:", len(target_lines_triggered))
