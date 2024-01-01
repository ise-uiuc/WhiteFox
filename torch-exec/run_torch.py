"""
This file runs the template_exec.py and restart it when it crashes.
"""
import subprocess as sp
import time
from pathlib import Path
import argparse
import os
import torch


def get_last_tested():
    tested_path = TEST_LOG_PATH
    if tested_path.exists() == False:
        return "start"

    text = tested_path.read_text()
    lines = text.splitlines()
    if len(lines) < 2:
        return "start"
    else:
        return lines[-2]


def combine_cov(cov_dir, cov_datafile):
    combine_cmds = [
        "coverage",
        "combine",
        f"--data-file={cov_datafile}",
        os.path.join(cov_dir, ".coverage.*"),
    ]
    output = sp.run(" ".join(combine_cmds), stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    if output.returncode != 0:
        print("combine coverage failed")
        print(output.stderr.decode())
        return


def collect_cov(cov_datafile):
    cov_jsonfile = cov_datafile.with_suffix(".json")
    ret = sp.run(
        [
            "python",
            "-m",
            "coverage",
            "json",
            f"--data-file={cov_datafile}",
            "-o",
            str(cov_jsonfile),
            "--pretty-print",
        ],
    )
    if ret.returncode != 0:
        print("collect coverage failed")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="the input dir of generated test cases",
    )
    parser.add_argument(
        "--res-dir",
        type=str,
        default="_results",
        help="the result dir to store outputs",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="the backend device to test"
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--cov", action="store_true", default=False, help="collect coverage"
    )
    parser.add_argument(
        "--cover",
        action="store_true",
        default=False,
        help="whether the inputs trigger/cover the optimization",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--validate", action="store_true", default=False)

    args = parser.parse_args()

    TIMEOUT = args.timeout
    DEVICE = args.device

    OUT_DIR = Path(args.input_dir)

    res_name = f"{OUT_DIR.name}-{args.device}"
    if args.suffix != "":
        res_name += f"-{args.suffix}"
    if args.validate:
        res_name += "-validate"
    RESULT_DIR = Path(args.res_dir) / res_name
    TEST_DIR = RESULT_DIR / "test"
    TEST_LOG_PATH = TEST_DIR / "tested.log"
    TEMP_LOG_PATH = TEST_DIR / "atemp.py"
    CRASH_LOG_PATH = TEST_DIR / "crash.log"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULT_DIR / "run.log"
    err_file = RESULT_DIR / "err.log"
    log_file.write_text("Start testing\n")
    err_file.write_text("Start testing\n")

    log_file = open(log_file, "a")
    stderr_file = open(err_file, "a")
    crash_file = open(CRASH_LOG_PATH, "a")
    timeout_file = open(RESULT_DIR / "timeout.log", "a")
    kill_file = open(RESULT_DIR / "killed.log", "a")
    start_time = time.time()

    # for coverage collection
    cov_cnt = 0
    cov_dir = Path(RESULT_DIR, "cov-datafile", "_cov_tmp_dir")
    cov_dir.mkdir(parents=True, exist_ok=True)
    cov_datafile = Path(RESULT_DIR, "cov-datafile", "my.coverage")

    while True:
        env = {
            "TORCHINDUCTOR_PERMUTE_FUSION": "1",
            "TORCHDYNAMO_VERBOSE": "1",
        }
        if not args.cover:
            env["TORCHINDUCTOR_SHAPE_PADDING"] = "1"
        env = {**env, **os.environ}

        if args.cov:
            cov_data = cov_dir / f".coverage.{cov_cnt}"
            commands = [
                "python",
                "-m",
                "coverage",
                "run",
                f"--source={torch.__path__[0]}",
                f"--data-file={cov_data}",
                "-a",
            ]
            cov_cnt += 1
        else:
            commands = ["python"]

        commands += [
            "template_exec.py",
            f"--out_dir={OUT_DIR}",
            f"--res_dir={RESULT_DIR}",
            f"--test_dir={TEST_DIR}",
            f"--test_log_path={TEST_LOG_PATH}",
            f"--temp_log_path={TEMP_LOG_PATH}",
            f"--device={DEVICE}",
        ]
        if args.validate:
            commands.append("--validate")
        if args.cover:
            commands.append("--cov")

        process = sp.Popen(
            commands,
            stdout=log_file,
            stderr=stderr_file,
            env=env,
        )

        count = 0
        while True:
            cur_test_target = get_last_tested()
            print("Current test target:", cur_test_target)
            result = process.poll()
            if result is None:
                time.sleep(1)
                if cur_test_target != get_last_tested():
                    cur_test_target = get_last_tested()
                    count = 0
                    continue
                elif count >= TIMEOUT:
                    print("TIMEOUT, Kill process")
                    process.kill()
                    timeout_file.write(f"{cur_test_target} TIMEOUT\n")
                    timeout_file.write(TEMP_LOG_PATH.read_text() + "\n")
                    timeout_file.flush()
                else:
                    count += 1
                    continue
            elif result == 233:
                print("FINISH")
                used_time = time.time() - start_time
                print("Used time:", used_time)

                if args.cov:
                    # combine the coverage
                    combine_cov(cov_dir, cov_datafile)
                    collect_cov(cov_datafile)

                # output_cov()
                log_file.write(f"\nUsed time: {used_time}")
                exit(233)
            elif result == 123:
                print("  Retrying ...")
            elif result == -9 or result == 255:
                # This is SIGKILL
                # We don't need to do anything
                kill_file.write(f"{cur_test_target} KILLED\n")
                kill_file.write(TEMP_LOG_PATH.read_text() + "\n")
                kill_file.flush()
                print(f"KILLED: {cur_test_target}")
            else:
                print(result)
                crash_file.write(
                    f"\n{cur_test_target} CRASH with return code {result}\n"
                )
                crash_file.write(TEMP_LOG_PATH.read_text() + "\n")
                crash_file.flush()
                print(f"ERROR: {cur_test_target}")
            break
        used_time = time.time() - start_time
        print("Restart at time:", used_time)
