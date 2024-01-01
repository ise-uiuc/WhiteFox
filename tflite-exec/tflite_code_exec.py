"""
Usage example:
RES_DIR="../output/starcoder/tflite/exec_results_0824/"
TRIGGER_INFO_PATH="${RES_DIR}/trigger/srcnl2test-feedback-iter2_trigger.jsonl"
echo '' > $TRIGGER_INFO_PATH
rm -r $RES_DIR

python tflite_code_exec.py --task_dir ../output/starcoder/tflite/srcnl2test-feedback-iter2/ --trigger_info_path ${TRIGGER_INFO_PATH} \
--res_dir=${RES_DIR} --test_dir=${RES_DIR} --test_log_path=${RES_DIR}/tested.log --temp_log_path=${RES_DIR}/temp_code.py 

"""

import argparse
import astunparse
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
from typing import List
from termcolor import colored
import re
import ast
import tempfile

import tensorflow as tf
import numpy as np
from tflite_oracle import ResType, run_tflite_oracle, tflite_test_executor_wrapper, TFLITE_TRIGGER_LOG_PATH
import traceback

TASK_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, TRIGGER_INFO_PATH = None, None, None, None, None, None, None
COV = False

OUTPUT_LIMIT: int = 1024
SEED: int = 420
MAXIMUM_TESTCASES = -1

# test_constant_init()
def read_all_tasks():
    tasks = []
    for opt_dir in TASK_DIR.iterdir():
        if not opt_dir.is_dir():
            continue
        opt = opt_dir.name
        cnt = 0
        for filename in opt_dir.iterdir():
            if not filename.name.endswith(".py"): continue

            label = str(filename)[:-3]
            code = filename.read_text()
            tasks.append([opt, label, code])

    tasks = sorted(tasks, key=lambda x: (x[0], x[1]))
    return tasks

def validate(code, filename):
    
    TEMP_LOG_PATH.write_text(code)

    # Get the current dir.
    cwd = os.getcwd()
    # Create a temp dir to execute the code
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Enter the temp dir to execute.
        os.chdir(tmpdirname)

        result, error, trigger_cnt = eval(f"tflite_test_executor_wrapper(run_tflite_oracle, code)")

        # Return to the previous dir
        os.chdir(cwd)

    if result == ResType.PASS:
        with open(TEST_DIR / 'success.log', 'a') as fw:
            fw.write(filename + '\n')
    else:
        with open(TEST_DIR / 'fail.log', 'a') as fw:
            fw.write(filename + '\n')

    return result, error, trigger_cnt

def core_oracle(code, filename, is_validate=False):
    result, error, trigger_cnt = validate(code, filename)
    info = {
        'filename': str(filename),
        'result': str(result),
        'error': error,
        'trigger_cnt': trigger_cnt,
    }
    with open(TRIGGER_INFO_PATH, 'a') as f:
        f.write(json.dumps(info) + '\n')

def core_loop(args):
    tasks = read_all_tasks()
    print(f'[INFO] Loaded {len(tasks)} tasks.')
    try:
        tested = set(open(TEST_LOG_PATH, 'r').read().splitlines())
    except Exception:
        tested = set([])

    count = 0
    for id in range(args.start, len(tasks)):
        task = tasks[id]
        api, label, code = task
        if args.target_task is not None:
            if label != args.target_task: continue
        filename = label + ".py"

        if filename in tested: continue
        with open(TEST_LOG_PATH, 'a') as fw:
            fw.write(filename + '\n')

        try:
            core_oracle(code, filename, is_validate=args.validate)
        except Exception as e:
            reason: str = "FrameworkCrashCatch"
            detail: str = str(e)
            if len(e.args) >= 2:
                reason: str = e.args[0]
                detail: str = e.args[1]
                
            if len(detail) > OUTPUT_LIMIT:
                detail = "Detail is too long"
            

            if reason == "FrameworkCrashCatch": # FrameworkCrashCatch is printed by driver
                print(traceback.format_exc())
                print(detail)
                exit(-1)

            if "Catch" in reason:
                with open("catches.log", "a") as f:
                    f.write("\nLmfuzzTestcase {} {} {} {} {} {}".format(id, api, label, reason, SEED, detail))
            print("\nLmfuzzTestcase", id, api, label, reason, SEED, detail)
            print("----------------------------------\n")

        count += 1
        if MAXIMUM_TESTCASES >= 0:
            if count >= MAXIMUM_TESTCASES:
                exit(123)



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_dir", type=str, default=None, help="The directory that contains the code to execute.")
    parser.add_argument("--res_dir", type=str, default=None, help="Path to store the results, like candidate bugs.")
    parser.add_argument("--trigger_info_path", type=str, default=None, help="Path to store the triggering results, a jsonl file.")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--test_log_path", type=str, default=None, help="Temp file to store the tested filenames.")
    parser.add_argument("--temp_log_path", type=str, default=None)
    parser.add_argument("--optim", type=str, default="inductor-post-grad")
    parser.add_argument("--validate", action="store_true", default=False)

    
    parser.add_argument("--target_task", type=str, default=None, help="A single task, for debug")

    parser.add_argument("--device", type=str, default="cpu")

    # for batch mode
    parser.add_argument("--start", type=int, default=0) # from which testcase to start
    parser.add_argument("--maximum_test_cases", type=int, default=-1) # 

    args = parser.parse_args()

    global TASK_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, COV, TRIGGER_INFO_PATH, MAXIMUM_TESTCASES
    TASK_DIR = Path(args.task_dir)
    RESULT_DIR = Path(args.res_dir)
    TEST_DIR = Path(args.test_dir)
    TEST_LOG_PATH = Path(args.test_log_path)
    TEMP_LOG_PATH = Path(args.temp_log_path)
    TRIGGER_INFO_PATH = Path(args.trigger_info_path)
    DEVICE = args.device

    RESULT_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)
    TRIGGER_INFO_PATH.parent.mkdir(exist_ok=True)

    MAXIMUM_TESTCASES = args.maximum_test_cases

    core_loop(args)

if __name__ == "__main__":
    main() 
    # Some sneaky code may contain exit(0) or other equivalent calls
    # We distinguish ourselves from them with a magic number
    exit(233)