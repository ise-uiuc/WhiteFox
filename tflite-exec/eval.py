"""Get triggering rate summary.

Usage: python tflite-exec/eval.py --trigger_data_path=output/tflite/feedback_loop_0907//trigger//step100-3shot-merged.json

Prints the summary:

Optim Name                                          # Trigger      Trigger(%)
FuseFullyConnectedAndAdd                                  848          84.80%
RemoveReshapeBeforeFullyConnected                         786          78.60%
RemoveReshapeAfterFullyConnected                          741          74.10%
FuseFullyConnectedAndMul                                  438          43.80%
FuseFullyConnectedAndReluX                                580          58.00%
ConvertTrivialTransposeOpToReshapeOp                      240          24.00%
FuseBinaryOpToFollowingAffineOp                           320          32.00%
FuseMulAndFullyConnected                                  457          45.70%
FuseUnpackAndConcatToReshape                              246          24.60%
FuseAddAndFullyConnected                                  316          31.60%
ScalarizeSplatConstantForBroadcastableOps                 264          26.40%
OptimizeTopK                                              229          22.90%

"""
import argparse
import json
import os
import pathlib

Path = pathlib.Path

def get_trigger_rate(trigger_data: dict):
    trigger_tests = trigger_data['trigger_tests']
    header_format = '{:<45} {:>15} {:>15}'
    headers = ['Optim Name', '# Trigger', 'Trigger(%)']
    print(header_format.format(*headers))
    trigger_cnt_total = 0
    trigger_rates = []
    row_format = '{:<45} {:>15} {:>15.2%}'
    for optim in trigger_tests:
        trigger_cnt = len(trigger_tests[optim])
        trigger_cnt_total += trigger_cnt
        trigger_rate = trigger_cnt / 1000
        trigger_rates.append(trigger_rate)
        print(row_format.format(optim, trigger_cnt, trigger_rate))
    trigger_rate_avg = sum(trigger_rates) / len(trigger_rates)
    print('{:<45} {:>15} {:>15.2}'.format(len(trigger_tests), trigger_cnt_total, trigger_rate_avg))

    

def get_trigger_number_from_jsonl(trigger_raw_jsonl_path: Path):
    data = trigger_raw_jsonl_path.read_text()
    data = data.splitlines()
    optim_trigger_test_cnt = {}
    trigger_test_cnt = 0
    for line in data:
        record = json.loads(line)
        trigger_cnt = record['trigger_cnt']
        if trigger_cnt:
            trigger_test_cnt += 1
        for optim in trigger_cnt:
            if optim not in optim_trigger_test_cnt:
                optim_trigger_test_cnt[optim] = 0
            optim_trigger_test_cnt[optim] += 1
        # if trigger_cnt:
        #     print(trigger_cnt)
        #     break
    print(optim_trigger_test_cnt)
    print('Trigger Optim', len(optim_trigger_test_cnt))
    print('Trigger Test', trigger_test_cnt)
    print('Total Test', len(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--trigger_data_path', type=str, help='Path of the json, containing the trigger data.')
    parser.add_argument('--trigger_raw_jsonl', type=str, default=None, help='Path of the raw jsonl, containing the trigger data.')
    parser.add_argument('--generation_batch_size', type=int, default=10, help='Default generations per step.')
    
    args = parser.parse_args()

    if args.trigger_raw_jsonl is not None:
        trigger_raw_jsonl_path = Path(args.trigger_raw_jsonl)
        get_trigger_number_from_jsonl(trigger_raw_jsonl_path)
    else:
        trigger_data_path = Path(args.trigger_data_path)
        if not os.path.exists(trigger_data_path):
            raise Exception(f'{trigger_data_path} does not exist.')
        get_trigger_rate(json.loads(trigger_data_path.read_text()))
        
        total_trigger_test_cnt = 0
        for i in range(1, 101):
            trigger_data_path = Path(args.trigger_data_path.replace('100', str(i)))
            data = json.loads(trigger_data_path.read_text())
            if 'newly_trigger_count' in data:
                for optim, value in data['newly_trigger_count'].items():
                    if 'target_opt_tests' in value:
                        total_trigger_test_cnt += value['target_opt_tests']
                        # print(i, value['target_opt_tests'])
        print('total_trigger_test_cnt', total_trigger_test_cnt)
        
    