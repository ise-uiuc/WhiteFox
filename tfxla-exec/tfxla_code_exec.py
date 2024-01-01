"""
Usage example:
RES_DIR="../output/starcoder/tfxla/exec_results_0824/"
TRIGGER_INFO_PATH="${RES_DIR}/trigger/srcnl2test-feedback-iter2_trigger.jsonl"
echo '' > $TRIGGER_INFO_PATH
rm -r $RES_DIR

python tfxla_code_exec.py --task_dir ../output/starcoder/tfxla/srcnl2test-feedback-iter2/ --trigger_info_path ${TRIGGER_INFO_PATH} \
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
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
from tfxla_oracle import ResType, run_tfxla_oracle, tfxla_test_executor_wrapper, TFXLA_TRIGGER_LOG_PATH, FAIL_MAPPING, NUM_MAPPING, nnsmith_run_tfxla_oracle
from tfxla_code_process import process_code
import traceback
import tf_code_process_titanfuzz
import run_nnsmith

TASK_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, TRIGGER_INFO_PATH = None, None, None, None, None, None, None
COV = False
RUNNING_TITANFUZZ = False
RUNNING_NNSMITH = False

OUTPUT_LIMIT: int = 1024
SEED: int = 420
MAXIMUM_TESTCASES = 100

# test_constant_init()

def clean_code(code: str) -> str:
    """Simple cleaning script. """
    code = code.strip()
    if code.endswith("```"):
        code = code[:-3]
    return code

def err_dict_to_str(err_dict):
    code = '\n'.join([f'**{k}**:\n{v}\n' for k, v in err_dict.items()])
    code = '"""\n' + code + '\n"""'
    return code

def read_all_tasks():
    tasks = []
    for opt_dir in TASK_DIR.iterdir():
        if not opt_dir.is_dir():
            continue
        opt = opt_dir.name
        if ('hangs' in opt) or ('seed' in opt): continue
        cnt = 0
        for filename in opt_dir.iterdir():
            if not filename.name.endswith(".py"): continue

            label = str(filename)[:-3]
            code = filename.read_text()
            # print(code)
            tasks.append([opt, label, code])

    tasks = sorted(tasks, key=lambda x: (x[0], x[1]))
    return tasks

def validate(code, filename):
    code = process_code(code)
    TEMP_LOG_PATH.write_text(code)

    # Get the current dir.
    cwd = os.getcwd()
    # Create a temp dir to execute the code
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Enter the temp dir to execute.
        os.chdir(tmpdirname)

        result, error, trigger_cnt = eval(f"tfxla_test_executor_wrapper(run_tfxla_oracle, code)")

        # Return to the previous dir
        os.chdir(cwd)

    if result in [ResType.AllPass, ResType.AllFail]: 
        with open(TEST_DIR / 'success.log', 'a') as fw:
            fw.write(filename + '\n')
    elif result != ResType.AllFail:                                               #TODO: filter out false positives
        with open(TEST_DIR / 'fail.log', 'a') as fw:
            fw.write(filename + '\n')
        result_name = str(result).replace('ResType.', '')
        result_dir = RESULT_DIR / result_name
        result_dir.mkdir(exist_ok=True)
        with open(result_dir / Path(filename).name, 'w') as fw:
            print(filename)
            fw.write(code + "\n" + err_dict_to_str(error) + "\n'''\n" + str(trigger_cnt) + "\n'''\n")

    return result, error, trigger_cnt

def validate_nnsmith(gen_path):
    # Get the current dir.
    cwd = os.getcwd()
    # Create a temp dir to execute the code
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Enter the temp dir to execute.
        os.chdir(tmpdirname)

        result, error, trigger_cnt = eval(f"tfxla_test_executor_wrapper(nnsmith_run_tfxla_oracle, gen_path)")

        # Return to the previous dir
        os.chdir(cwd)
    
    filename = str(gen_path)
    if result in [ResType.AllPass, ResType.AllFail]: 
        with open(TEST_DIR / 'success.log', 'a') as fw:
            fw.write(filename + '\n')
    elif result != ResType.AllFail:                                               #TODO: filter out false positives
        with open(TEST_DIR / 'fail.log', 'a') as fw:
            fw.write(filename + '\n')
        result_name = str(result).replace('ResType.', '')
        result_dir = RESULT_DIR / result_name
        result_dir.mkdir(exist_ok=True)
        with open(result_dir / Path(filename).name, 'w') as fw:
            print(filename)
            fw.write(filename + "\n" + err_dict_to_str(error) + "\n'''\n" + str(trigger_cnt) + "\n'''\n")

    return result, error, trigger_cnt

def core_nnsmith_oracle(gen_path):
    result, error, trigger_cnt = validate_nnsmith(gen_path)
    info = {
        'filename': str(gen_path),
        'result': str(result),
        'error': error,
        'trigger_cnt': trigger_cnt,
    }
    with open(TRIGGER_INFO_PATH, 'a') as f:
        f.write(json.dumps(info) + '\n')

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
    if RUNNING_NNSMITH:
        tasks = run_nnsmith.read_all_nnsmith_tasks(TASK_DIR)
    else:
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
        
        if RUNNING_NNSMITH:
            filename = label
        else:
            filename = label + ".py"
        print(filename)

        if filename in tested: continue
        with open(TEST_LOG_PATH, 'a') as fw:
            fw.write(filename + '\n')

        if not RUNNING_NNSMITH:
            code = clean_code(code)
        if RUNNING_TITANFUZZ:
            code, _ = tf_code_process_titanfuzz.TFCodeGenerator.generate(code)
        try:
            if RUNNING_NNSMITH:
                core_nnsmith_oracle(Path(filename))
            else:
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
    parser.add_argument("--titanfuzz", action="store_true", default=False)
    parser.add_argument("--nnsmith", action="store_true", default=False)

    
    parser.add_argument("--target_task", type=str, default=None, help="A single task, for debug")

    parser.add_argument("--device", type=str, default="cpu")

    # for batch mode
    parser.add_argument("--start", type=int, default=0) # from which testcase to start
    parser.add_argument("--maximum_test_cases", type=int, default=-1) # 

    args = parser.parse_args()

    global TASK_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, COV, TRIGGER_INFO_PATH, MAXIMUM_TESTCASES
    global RUNNING_TITANFUZZ, RUNNING_NNSMITH
    TASK_DIR = Path(args.task_dir)
    RESULT_DIR = Path(args.res_dir)
    TEST_DIR = Path(args.test_dir)
    TEST_LOG_PATH = Path(args.test_log_path)
    TEMP_LOG_PATH = Path(args.temp_log_path)
    TRIGGER_INFO_PATH = Path(args.trigger_info_path)
    DEVICE = args.device
    RUNNING_TITANFUZZ = args.titanfuzz
    RUNNING_NNSMITH = args.nnsmith

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