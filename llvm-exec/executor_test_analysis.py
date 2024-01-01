import json
import os
import subprocess
import os
import time
from pathlib import Path
from executor_test import info_read


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
    file: str, out_dir="/JawTitan/whitefox-data/starcoder/llvm-exe-result/"
):
    """
    llc compile error
    linker linker error
    exe error
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
    file_cpp = os.path.join(out_folder, "file.cpp")
    file_before_opt = os.path.join(out_folder, "file_before_opt.ll")
    file_after_opt = os.path.join(out_folder, "file_after_opt.ll")
    exe_before_opt = os.path.join(out_folder, "exe_before_opt")
    exe_after_opt = os.path.join(out_folder, "exe_after_opt")

    commands = {
        "cp_ori_cc": ["cp", str(file).replace(".ll", ""), file_cpp],
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

    error_dict = {}

    for k, v in commands.items():
        # print(" ".join(v))
        print(" ".join([str(item) for item in v]))
        k = str(k)
        try:
            result = subprocess.run(commands[k], capture_output=True)
            _, filename = os.path.split(commands[k][-1])
            print(result.returncode)

            error_dict[k] = {
                "returncode": result.returncode,
            }
            if result.returncode == 0:
                continue
            with open(os.path.join(out_folder, filename + ".err"), "w") as f:
                print(result.stderr)
                f.write(result.stderr.decode())
            error_dict[k]["stderr"] = result.stderr.decode()
        except subprocess.CalledProcessError as e:
            print(None, e.stderr, e.returncode)
            continue
        except Exception as e:
            print(e)
            continue

    return error_dict


def filter_error_dict(error_dict: dict):
    for k, v in error_dict.items():
        if v["returncode"] != 0:
            return False
    return True


if __name__ == "__main__":
    # python executor_test_analysis.py --diff-json /JawTitan/whitefox-data/starcoder/llvm-exe-result/diff_test_mixed_nl/diff_result.json
    # python executor_test_analysis.py --diff-json /JawTitan/whitefox-data/starcoder-1000/llvm-exe-result/diff_test_llvm_opt_run/diff_result.json

    diff_type = ["llc", "linker", "exe"]

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diff-json",
        type=str,
        default="/JawTitan/whitefox-data/starcoder/llvm-exe-result/diff_test1/diff_result.jsonl",
    )
    args = parser.parse_args()

    data = info_read(args.diff_json)

    diff_file = {
        "llc": set(),
        "linker": set(),
        "exe": set(),
    }
    cnt = 0
    for k1, v1 in data.items():
        for type in diff_type:
            try:
                if (
                    v1[f"{type}_before"]["returncode"]
                    != v1[f"{type}_after"]["returncode"]
                ):
                    if v1[f"{type}_before"]["returncode"] == -6:
                        continue
                    print(
                        f"{type} diff, ",
                        k1,
                        "before, ",
                        v1[f"{type}_before"]["returncode"],
                        "after, ",
                        v1[f"{type}_after"]["returncode"],
                    )
                    diff_file[type].add(k1)
                if type == "exe":
                    cnt += 1
            except:
                continue

    for k, v in diff_file.items():
        print(k, len(v))

    out_dir = Path("bug-analysis")
    out_dir.mkdir(exist_ok=True)
    for k, files in diff_file.items():
        k_out_dir = out_dir / k
        k_out_dir.mkdir(exist_ok=True)
        log_file = k_out_dir / "log.txt"
        for file in diff_file[k]:
            error_dict = test_executor(file, k_out_dir)
            with open(log_file, "a+") as f:
                f.write(f"{file}\n")
                f.write(json.dumps(error_dict, indent=4))
                f.write("\n")
                f.write("-" * 100)
