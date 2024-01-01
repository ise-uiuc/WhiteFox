import json
import argparse
import os

from json_parser import traversal, extract_source_from_llvm, extract_func_body_from_source, gen_prompt
from add_instrument import add_instrument
from utils import statistics

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--json", type=str, default="example.json")
	parser.add_argument("--llvm-source", type=str, default="")
	args = parser.parse_args()
	json_file = args.json
	with open(json_file, "r") as file:
		data = json.load(file)
		

	# change the path
	for k in data.keys():
		for idx, _ in enumerate(data[k]["hints"]):
			data[k]["hints"][idx]["codes"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][0])
			try:
				data[k]["hints"][idx]["codes"][1] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][1])
			except:
				pass
			data[k]["hints"][idx]["examples"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["examples"][0])
			data[k]["hints"][idx]["specific_ir"] = os.path.join(args.llvm_source, data[k]["hints"][idx]["specific_ir"])

	''' old api deprecated'''
	# traversal(data)

	''' new api, you can refer to the readme.md or the docstring of every func '''
	extract_source_from_llvm(data) # recommend to comment this line if you only want to gen the prompt
	extract_func_body_from_source(data)
	# gen_prompt(data, "cpp")
	# gen_prompt(data, "ll")
	# gen_prompt(data, "src2nl_gpt4")
	# gen_prompt(data, "no_source")
	# gen_prompt(data, "no_source_with_specific_ir")
	# gen_prompt(data, "starcoder-src2nl-tutorial")
	# gen_prompt(data, "nl_pattern")
	# gen_prompt(data, "cpp_deadarg")
	# gen_prompt(data, "cpp_aggrinst")
	# gen_prompt(data, "starcoder_cpp_deadarg")

	# add instrument to llvm_project
	add_instrument(data) # recommend to comment this line if you only want to gen the prompt

	# simple data statistics
	statistics(data)
