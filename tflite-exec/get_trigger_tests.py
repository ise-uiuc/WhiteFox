"""
RES_DIR="../output/starcoder/tflite/exec_results_0824/"
TRIGGER_INFO_PATH="${RES_DIR}/trigger/srcnl2test-feedback-iter2_trigger.jsonl"
TRIGGER_MERGED_PATH="${RES_DIR}/trigger/srcnl2test-feedback-iter2_trigger_step1_merged.jsonl"
python get_trigger_tests.py --trigger_info_path=${TRIGGER_INFO_PATH} --trigger_merged_path=$TRIGGER_MERGED_PATH
"""
import argparse
import copy
import json
from pathlib import Path

def extract_code(trigger_tests):
    ret = dict()
    for opt, file_list in trigger_tests.items():
        test_list = []
        codes = set()
        for file_name in file_list:
            code = Path(file_name).read_text()
            if code not in codes:
                test_list.append({
                    "code": code,
                    "alpha": 1,
                    "beta": 1,
                })
                codes.add(code)
        ret[opt] = test_list
    return ret


def get_trigger_info(trigger_info_path: Path):
    """Extract trigger information from raw data.
    
    The trigger_info follows this format:
    'trigger_tests': {
        'RemoveReshapeBeforeFullyConnected': [
            {
                'code': 'class Model( ...',
                'alpha': 1,
                'beta': 1,
            },
            ...
        ],
        ...
    }
    'newly_trigger_count': {
        'RemoveReshapeBeforeFullyConnected': {
            'all_tests': 2,
            'target_opt_tests': 1
        },
        ...
    }
    'RemoveReshapeBeforeFullyConnected_selected': [0, 1, 2],
    
    """
    data = trigger_info_path.read_text().splitlines()
    data = [json.loads(line) for line in data]
    trigger_info = dict()
    trigger_tests = dict()
    newly_trigger_count = dict()
    for record in data:
        file_name = record['filename']
        if file_name.startswith('../'):
            file_name = file_name[3:]
        trigger_cnt: dict = record['trigger_cnt']
        for opt, opt_trigger_cnt in trigger_cnt.items():
            if opt_trigger_cnt == 0: continue
            if opt not in trigger_tests:
                trigger_tests[opt] = []
            if opt not in newly_trigger_count:
                newly_trigger_count[opt] = {
                    'all_tests': 0,
                    'target_opt_tests': 0
                }
            trigger_tests[opt].append(file_name)
            newly_trigger_count[opt]['all_tests'] += 1
            
            if Path(file_name).name.startswith(opt):
                # If the newly triggered tests comes from the generation for the target opt.
                newly_trigger_count[opt]['target_opt_tests'] += 1
    
    trigger_info['trigger_tests'] = extract_code(trigger_tests)
    trigger_info['newly_trigger_count'] = newly_trigger_count
    return trigger_info


def merge_trigger_info(trigger_info, prior_trigger_info, generation_batch_size:int):
    # If prior trigger info is empty, return the current info.
    if 'trigger_tests' not in prior_trigger_info:
        return trigger_info
    merged_trigger_info = copy.deepcopy(prior_trigger_info)
    # Append new triggering tests.
    trigger_tests = trigger_info['trigger_tests']
    for opt, test_list in trigger_tests.items():
        if not opt in merged_trigger_info['trigger_tests']:
            merged_trigger_info['trigger_tests'][opt] = test_list
            continue
        prior_test_list = merged_trigger_info['trigger_tests'][opt]
        codes = set([test["code"] for test in prior_test_list])
        for test in test_list:
            code = test["code"]
            if code not in codes:
                merged_trigger_info['trigger_tests'][opt].append(test)
        
    # Update alpha, beta
    for opt_sel_label, opt_selected_indices in prior_trigger_info.items():
        if not opt_sel_label.endswith('_selected'): continue
        opt = opt_sel_label.replace('_selected', '')
        
        # Get the generation success information
        num_success = 0
        if opt in trigger_info['newly_trigger_count']:
            num_success = trigger_info['newly_trigger_count'][opt]['target_opt_tests']
        num_fail = generation_batch_size - num_success

        for id in opt_selected_indices:
            merged_trigger_info['trigger_tests'][opt][id]['alpha'] += num_success
            merged_trigger_info['trigger_tests'][opt][id]['beta'] += num_fail
    return merged_trigger_info
             

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--code_dir", type=str, default="The directory that contains the generated code.")
    parser.add_argument("--trigger_info_path", type=str, help="Path of the jsonl, containing the trigger data.")
    parser.add_argument("--trigger_prior_path", type=str, default=None, help="Path of the previous trigger data.")
    parser.add_argument("--trigger_merged_path", type=str, help="Path of the output merged trigger data.")
    parser.add_argument("--generation_batch_size", type=int, default=10, help="Default generations per step.")
    
    args = parser.parse_args()
    
    # Load the latest trigger info
    trigger_info = get_trigger_info(Path(args.trigger_info_path))
    print(len(trigger_info))
    # Merge with prior
    generation_batch_size = args.generation_batch_size
    if args.trigger_prior_path is None or not Path(args.trigger_prior_path).exists():
        merged_trigger_info = trigger_info
    else:
        prior_trigger_path = Path(args.trigger_prior_path)
        prior_trigger_info = json.loads(prior_trigger_path.read_text())
        merged_trigger_info = merge_trigger_info(trigger_info, prior_trigger_info, generation_batch_size)

    trigger_merged_path = Path(args.trigger_merged_path)
    trigger_merged_path.parent.mkdir(exist_ok=True)

    json.dump(merged_trigger_info, trigger_merged_path.open('w'), indent=4)
    
