"""
The script to evaluate the match coverage.
"""

from pathlib import Path
from collections import defaultdict
import json
import argparse
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'exec'))
from template_exec import CodeParser

# Get the mappings
mapping_file = Path('eval/match_mapping.txt')
match_mapping = {}
for line in mapping_file.read_text().splitlines():
    if "inner" not in line and "inplace" not in line:
        if line.split(": ")[1] in match_mapping:
            print("duplicate key: ", line.split(": ")[0], match_mapping[line.split(": ")[1]])
        match_mapping[line.split(": ")[1]] = line.split(": ")[0]

output_keys = [
    "fuse_conv_bn",
    "linear_permute_fusion",
    "permute_linear_fusion",
    "permute_matmul_fusion",
    "replace_fx",
    "sink_cat_after_pointwise",
    "addmm",
    "cat_addmm",
    "cat_mm",
    "cat_slice_cat",
    "mm_plus_mm",
    "pointless_cumsum_replacement",
    "splitwithsizes_cat_replace",
    "sfdp=0",
    "sfdp=1",
    "sfdp=2",
    "sfdp=3",
    "sfdp=4",
    "sfdp=5",
] + list(match_mapping.values())

mappings = {
    "sfdp=0": "_sfdp_replacement_1",
    "sfdp=1": "_sfdp_replacement_2",
    "sfdp=2": "_sfdp_replacement_3",
    "sfdp=3": "_sfdp_replacement_4",
    "sfdp=4": "_sfdp_replacement_5",
    "sfdp=5": "_sfdp_replacement_6",
}

def replace(name):
    for k, v in mappings.items():
        if v in name:
            print(name)
            name = name.replace(v, k)
            print(name)
    return name

def convert_to_dict(file_path):
    if not file_path.exists(): return {}
    lines = file_path.read_text().splitlines()
    res = {}
    for l in lines:
        data = json.loads(l)
        for k, cov_list in data.items():
            max_cov = set()
            for cov in cov_list:
                if cov in match_mapping.keys():
                    cov = match_mapping[cov]
                    # For the other cases not in the match_mapping, keep the original name.

                # unary=3 in unary=3_0.py --> it is in
                if k.startswith(replace(cov)):
                    max_cov.add(cov)
            res[k] = max_cov 
    return res

def merge_cover(cov1, cov2):
    for k, _ in cov2.items():
        if k in cov1:
            cov1[k] = cov1[k].union(cov2[k])
        else:
            cov1[k] = cov2[k]
    return cov1

def extract_code(trigger_info, optim_init, code_parser):
    output = {}
    for opt, file_list in trigger_info.items():
        # print(opt)
        opt_list = []
        existings = set()
        total = 0
        unique = 0
        for file_name in file_list:
            origin_code = Path(file_name).read_text()

            try:
                model_code, _, input_code = code_parser.split_func_tensor(origin_code)
            except Exception as e:
                print(origin_code)

            total += 1
            if (model_code, input_code) not in existings: 
                # TODO: change the alpha and beta here.
                opt_list.append({
                    "model_code": model_code,
                    "input_code": input_code,
                    "alpha": optim_init[opt][0],
                    "beta": optim_init[opt][1],
                })
                existings.add((model_code, input_code))
                unique += 1
        # print(total)
        # print(len(opt_list))
        output[opt] = opt_list
    return output

def add_trigger_file(trigger_files: dict, name, filename):
    if name not in trigger_files:
        trigger_files[name] = set()
    trigger_files[name].add(filename)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="")
    args.add_argument("--base_name", type=str, default="base.json")
    args.add_argument("--batch_size", type=int, default=10)
    args.add_argument("--data_dir", type=str, default="_results-new") # FIXME: change this
    args.add_argument("--code_dir", type=str, default="chatgpt-tests")
    args.add_argument("--out_dir", type=str, default="match-data")

    code_dir = args.parse_args().code_dir
    data_dir = args.parse_args().data_dir
    data_name = args.parse_args().data_name

    out_dir = Path(args.parse_args().out_dir)
    out_dir.mkdir(exist_ok=True)
    output_file = out_dir / f"{data_name}.txt"
    output_file = output_file.open('w')

    # specify the json datafile
    data_file_cpu = Path(data_dir, f"{data_name}-cpu", 'test', 'match.log')
    data_file_cuda = Path(data_dir, f"{data_name}-cuda", 'test', 'match.log')
    
    code_path = Path(code_dir, f"{data_name}")

    trigger_info_path = out_dir / f"{data_name}.json"
    merged_info_path = out_dir / f"{data_name}-merged.json"

    covered_cpu = convert_to_dict(data_file_cpu)
    covered_cuda = convert_to_dict(data_file_cuda)

    ans = merge_cover(covered_cpu, covered_cuda)
    # [opt_name, triggered_file_names]
    triggered_files: dict[str, set] = {}
    # [opt_name, (avg_alpha, avg_beta)]
    optim_init: dict[str, tuple[int, int]] = {opt: (1, 1) for opt in output_keys}

    per_cnt = defaultdict(int)
    tot_cnt = defaultdict(int)
    for k, v in ans.items():
        for name in v:
            name = replace(name)
            if name not in output_keys: continue
            tot_cnt[name] += 1
            if k.startswith(name):
                per_cnt[name] += 1
            filename = code_path / "_".join(k.split("_")[:-1]) / k
            add_trigger_file(triggered_files, name, str(filename))
            

    tot_cover = 0
    tot_cover_cnt = 0
    for v in output_keys:
        if v.startswith("inplace"): continue
        if v in per_cnt:
            output_file.write(f"{v}, {per_cnt[v]}\n")
            tot_cover += 1
            tot_cover_cnt += per_cnt[v]
        else:
            output_file.write(f"{v}, 0\n")

    print(tot_cover, tot_cover_cnt)
    print(tot_cover, tot_cover_cnt, file=sys.stderr)
    output_file.write("-----------------\n")
    print(len(tot_cnt))

    for v in output_keys:
        if v.startswith("inplace"): continue
        if v in per_cnt:
            output_file.write(f"{v}, {tot_cnt[v]}\n")
    
    # Merge the new trigger info with the old one.
    # base_file is the trigger info from the previous run.
    base_file = Path(args.parse_args().base_name)
    batch_size = args.parse_args().batch_size
    merged_trigger_codes = {}
    if base_file.exists():
        base_trigger_codes = json.load(base_file.open('r'))
        # update the alpha and beta
        for optim in output_keys:
            if f"{optim}_selected" not in base_trigger_codes: continue
            seleceted_list = base_trigger_codes[f"{optim}_selected"]
            avg_alpha = 0
            avg_beta = 0
            for idx in seleceted_list:
                if "alpha" not in base_trigger_codes[optim][idx]:
                    base_trigger_codes[optim][idx]["alpha"] = 1
                if "beta" not in base_trigger_codes[optim][idx]:
                    base_trigger_codes[optim][idx]["beta"] = 1
                base_trigger_codes[optim][idx]["alpha"] += per_cnt[optim]
                base_trigger_codes[optim][idx]["beta"] += batch_size - per_cnt[optim]
                avg_alpha += base_trigger_codes[optim][idx]["alpha"]
                avg_beta += base_trigger_codes[optim][idx]["beta"]
            # Dynamic alpha and beta based on the chosen ones.
            avg_alpha //= len(seleceted_list)
            avg_beta //= len(seleceted_list)
            optim_init[optim] = (avg_alpha, avg_beta)
    else:
        base_trigger_codes = {}

    code_parser = CodeParser()
    trigger_codes = extract_code(triggered_files, optim_init, code_parser)
    json.dump(trigger_codes, trigger_info_path.open('w'), indent=4)

    # Merge the new trigger info with the old one.
    for optim in output_keys:
        if optim not in trigger_codes and optim not in base_trigger_codes: continue
        if optim not in trigger_codes:
            merged_trigger_codes[optim] = base_trigger_codes[optim]
        elif optim not in base_trigger_codes:
            merged_trigger_codes[optim] = trigger_codes[optim]
        else:
            merged_trigger_codes[optim] = base_trigger_codes[optim] + trigger_codes[optim]
    json.dump(merged_trigger_codes, merged_info_path.open('w'), indent=4)
