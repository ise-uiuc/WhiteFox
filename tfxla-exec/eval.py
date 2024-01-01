# import tensorflow as tf
import numpy as np
from utils import collect_trigger_info, xla_run
from pathlib import Path
import argparse

#FIXME: Needs adapting to new path (name) changes

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="outputs")
parser.add_argument("--mode", type=str, default="xla", choices= ['xla','autocluster','naive'])
parser.add_argument("--optim", type=str, default="dynamic_dimension_simplifier")
args = parser.parse_args()

optim_dir = Path(args.dir)
optim_name = args.optim+".cc"
mode = args.mode

TFXLA_TRIGGER_LOG_PATH = "/tmp/xla_trigger.log"
TFXLA_TRIGGER_INFO_PATH = f"{mode}_trigger_summary.txt"
TFXLA_VALID_INFO_PATH = f"{mode}_valid_summary.txt"

batch_size = len(optim_dir.iter_dir)
trigger_cnt = 0
valid_cnt = 0
code_unique = []

for src_dir in optim_dir.iterdir():
    print(str(src_dir))
    code = src_dir.read_text()
    if code in code_unique:
        code_unique.append(code)
    else:
        print("Duplicated.")
    Path(TFXLA_TRIGGER_LOG_PATH).write_text('')
    try:
        ouput = xla_run(code, mode)
        valid_cnt += 1
    except:
        print("Run failed.")
        continue
    if collect_trigger_info(TFXLA_TRIGGER_LOG_PATH)[optim_name]:
        print("Optimization successfully triggered!")
        trigger_cnt += 1
    else:
        print("Optimization not triggered.")

#FIXME: Needs filling the following two `write_text`

Path(TFXLA_TRIGGER_INFO_PATH).write_text()
Path(TFXLA_VALID_INFO_PATH).write_text()