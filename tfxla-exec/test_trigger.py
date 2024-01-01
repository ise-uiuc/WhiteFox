import tensorflow as tf
import numpy as np
from utils import collect_trigger_info, xla_run
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="tests/test_dot_merger.py")
parser.add_argument("--mode", type=str, default="xla", choices= ['xla','autocluster','naive'])
parser.add_argument("--optim", type=str, default="dynamic_dimension_simplifier")
args = parser.parse_args()

src_dir = Path(args.dir)
optim_name = args.optim+".cc"

#print(optim_name)

TFXLA_TRIGGER_LOG_PATH = "/tmp/xla_trigger.log"
Path(TFXLA_TRIGGER_LOG_PATH).write_text('')


code = src_dir.read_text()
ouput = xla_run(code, args.mode)
if collect_trigger_info(TFXLA_TRIGGER_LOG_PATH)[optim_name]:
    print("Optimization successfully triggered!")
else:
    print("Optimization not triggered.")