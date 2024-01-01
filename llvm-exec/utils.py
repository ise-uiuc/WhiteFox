import os
import shutil

def extract_function_body(cpp_file, function_name):
	with open(cpp_file, 'r') as file:
		lines = file.readlines()
	start_line = -1
	end_line = -1
	in_function = False

	count_braces = 0
	for i, line in enumerate(lines):
		if line.strip().startswith('PreservedAnalyses ' + function_name) or (function_name in line):
			start_line = i
			in_function = True
			func_begin = True
		if in_function:
			count_braces += line.count("{")
			if func_begin and ';' in line:
				in_function = False
				func_begin = False
				continue
			if func_begin and count_braces == 0:
				continue
			else:
				func_begin=False
			count_braces -= line.count("}")
			if "}" in line:
				end_line = i
			if count_braces == 0 and i != start_line:
				break

	if start_line != -1 and end_line != -1:
		function_body = ''.join(lines[start_line:end_line+1])
		return function_body
	else:
		return None

def simplify_func(func_list:list, func_body_list:list, target_line:str, target_dict:dict, outfile:str=None):
	'''
	the func we get is too long and contain too much target_lines, so we simplify them
	'''

	'''
	The llvm source codes are well orgnized, 
	so I may as well think every block is split by the blank line :)
	'''

	func_body_list_after = []
	
	# remove the redundant `return false;`
	for func_name, func_body in zip(func_list, func_body_list):
		print(func_name)
		paragraphs = func_body.split('\n\n')  # Split on blank lines

		if "return false;" in paragraphs[0]:
			paragraphs = [list(paragraphs[0].split('{'))[0] + "{", list(paragraphs[0].split('{'))[1]] + paragraphs[1:]
		
		# before write into the file, we should merge the block like for-loop into one block, 
		# and it is convient for us to remove other target line
		paragraph_after_list = merge_block(paragraphs)
		func = find_right_func(target_dict, target_line)
		if func != func_name:
			# we get into a higher func
			# print(list(list(func.split("::"))[-1].split("("))[0])
			# print([item for item in target_dict.keys() if item != func])
			if " " in func:
				target = list(list(func.split("("))[0].split(" "))[-1]
			else:
				target = func
			if "::" in target:
				target = list(list(target.split("::"))[-1].split("("))[0]
			# print(f"func name is {func}")
			print(f"extract func name is {target}")
			paragraph_after_list = remove_block(paragraph_after_list , target, [list(list(item.split("::"))[-1].split("("))[0] for item in target_dict.keys() if item != func])
			tmp = [paragraph_after_list[0]]
			for idx, para in enumerate(paragraph_after_list):
				if idx == 0:
					continue
				if target in para or 'return' in para or 'false' in para:
					tmp.append(para)
			if paragraph_after_list[-1] not in tmp:
				tmp.append(paragraph_after_list[-1])
			paragraph_after_list = tmp
		else:
			# we get into this func
			# print([item for item in target_dict[func] if item != target_line])
			print(f"extract target line is {target_line}")
			paragraph_after_list = remove_block(paragraph_after_list, target_line, [item for item in target_dict[func] if item != target_line], True)
	
		# paragraph_after_list = []
		# for para in paragraphs:
		# 	if "return false;" in para and "Changed" not in para and "return true" not in para:
		# 		# continue
		# 		paragraph_after_list.append(para)
		# 	else:
		# 		continue
		# 		paragraph_after_list.append(para)
		

		func_body_after = "\n\n".join(paragraph_after_list)

		func_body_list_after.append(func_body_after)
		func_body_list_after.append("\n")
	
	if outfile is not None:
		# only for debug
		write_onelist2file(func_body_list_after, outfile)
	
		# # remove the comment
		# content = remove_commented_lines(outfile, "/")
		# with open(outfile, "w") as file:
		# 	file.write(content)

	return func_body_list_after
	pass

def remove_block(para_list:list, target:str, others:list, judge1=False):
	# if len(others) == 0:
		# return para_list
	
	def judge1(item:str, target:str, others:list):
		if target in item:
			return True
		if 'return false' in item or 'return true' in item or 'return Changed' in item or 'return nullptr' in item:
			return True
		for other in others:
			if other in item:
				return False
		return False
		pass

	def judge(item:str, target:str, others:list):
		if target in item:
			return True
		for other in others:
			if other in item:
				return False
		return True
		pass
	
	if judge1 is False:
		new_list = [item for item in para_list if judge(item, target, others)]
	else:
		new_list = [item for item in para_list if judge1(item, target, others)]

	if para_list[-1] not in new_list:
		new_list.append(para_list[-1])

	if para_list[0] not in new_list:
		new_list = [para_list[0]] + new_list

	return new_list
	pass


def find_right_func(target_dict:dict, target_line):
	for k, v in target_dict.items():
		if target_line in v:
			return k
	return None
	pass

def merge_block(para_list:list):
	'''
	Accroding the indent size, we merge some blocks into one.
	'''
	para_list = [item.strip("\n") for item in para_list]
	new_list = [para_list[0].rstrip('\n')]
	now_block_idx = 1
	for idx, para in enumerate(para_list):
		if idx < now_block_idx:
			continue
		new_para = para
		now_block_idx += 1
		# print(count_block_indent_min(para))
		while now_block_idx < len(para_list) and count_block_indent_min(para_list[now_block_idx]) > count_block_indent_min(para):
			# print(count_block_indent_min(para))
			new_para += para_list[now_block_idx] + '\n\n'
			now_block_idx+=1

		if now_block_idx != (idx + 1) and now_block_idx < len(para_list):
			new_para += para_list[now_block_idx] + '\n\n'
			now_block_idx+=1

		new_list.append(new_para)
		# new_list.append(new_para + "\n\n\n1\n\n\n")

	# print(para_list)
	# for para in para_list:
	# 	print(para)
	# 	print("1")

	# for para in new_list:
	# 	print(para)
	# 	print("11111")

	return new_list
	pass

def count_block_indent_min(block: str):
	return max(min([count_prefix_indent(line) for line in block.split("\n")]), 2)
	pass

def count_prefix_indent(text):
    return len(text) - len(text.lstrip())

def select_target_line(all_target_lines:dict, target_line:str, func_name:str):
	pass

def write_onelist2file(onelist:list, filename:str):
	with open(filename, "w") as file:
		for item in onelist:
			if item is not None:
				file.write(item)

def create_file(file_name: str):
	'''
	ugly code :(
	this is a function that just create a new file, I don't know more efficient methods.
	'''
	if os.path.exists(file_name):
		return None
	# Open a file in write mode
	file = open(file_name, "x")
	# Close the file
	file.close()

	# import subprocess
	# command = f"touch {file_name}"
	# subprocess.run(command, shell=True, capture_output=True, text=True)

def copy_file(source_file, dest):
	'''
	source_file: the file of source, a file
	dest: directory
	'''
	# source_file = pass_info["codes"][0] # TODO Only first cpp file!
	# dest ="source-code-data/llvm/llvm-lib"
	dest_file = os.path.join(dest, os.path.basename(source_file))
	if not os.path.exists(dest_file):
		shutil.copy(source_file, dest)
	return dest_file

def remove_commented_lines(file_path, comment_symbol=';'):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove lines that start with a semicolon
    lines = [line for line in lines if not line.strip().startswith(comment_symbol)]

    # Join the lines back into a single string
    content = ''.join(lines)

    return content

def remove_extra_blank_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    with open(filename, 'w') as file:
        prev_line = ''
        for line in lines:
            line = line.rstrip('\n')
            if line.strip() == '' and prev_line.strip() == '':
                continue
            file.write(line + '\n')
            prev_line = line

import re
def find_passes_from_ll_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

	# 正则表达式模式
    pattern = r"-passes=(\S+)"
    # pattern = r'(?<=; RUN: llc < %s).*?(?=\| FileCheck %s)'
    # pattern = r'; RUN: llc.*\| FileCheck.*?; (.*?) ;'

	# 使用正则表达式进行匹配
    matches = re.findall(pattern, content)
    # matches = re.search(pattern, content)

    if matches:
        # Remove the single quotes or double quotes from the match
        return matches[0]
        # extracted_content = matches.group(1).strip()
        # print(extracted_content)
    else:
        return None

def statistics(data:dict):
	num = 0
	for k, passes in data.items():
		num += len(passes["hints"])
	print(f"We has generated {num} prompt files")
	print(f"We has collected {len(data.keys())} optimization")


def has_main_function(ll_code):
    # Check if the .ll code contains a definition for the main function
    return "define i32 @main(" in ll_code

def add_main_function(ll_code):
    # Split the .ll code into lines
    lines = ll_code.split("\n")

    # Find all function names defined in the .ll code along with their parameters
    functions = [line for line in lines if line.startswith("define")]
    function_names = [func.split(" ")[2] for func in functions]

    # params define
    params_define = []

    # Create the main function that calls all other functions
    main_function = "define i32 @main() {\n"
    # for func_line in functions:

    #     # regulary expr
    #     "%variable_name = alloca data_type"
    #     import re
    #     params = re.findall(r"\((.*?)\)", func_line)
    #     params = list(params[0].split(","))
    #     print(params)
    #     for param in params:
    #         param = param.strip(" ").split(" ")
    #         print(param)
    #         alloca = f"  {param[1]} = alloca {param[0]}\n"
    #         if alloca in params_define:
    #             continue
    #         main_function += alloca
    #         params_define.append(alloca)

    #     # invoke call in function
    #     replacement = func_line.replace("define", "call").strip("{")
    #     main_function += f"  {replacement}\n"

    main_function += "  ret i32 0\n}\n"

    # Append the main function to the .ll code
    lines.append(main_function)

    # Join the lines back into a single string
    updated_ll_code = "\n".join(lines)

    return updated_ll_code

def update_ll_file(file_path:str, out_file:str):
    # Read the .ll file
    ll_code = remove_commented_lines(file_path)

    # Check if the .ll code has a main function
    if not has_main_function(ll_code):
        # Add the main function
        updated_ll_code = add_main_function(ll_code)

        # Write the updated .ll code back to the file
        # with open(file_path, "w") as file:
        with open(out_file, "w") as file:
            file.write(updated_ll_code)
        
        remove_extra_blank_lines(out_file)
        return True

    return False

def count_characters(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            character_count = len(content)
            return character_count
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import re
def extract_code_from_markdown(markdown_text, code="llvm"):
    pattern = fr'```{code}\s*\n(.*?)```'
    # pattern = fr'```{code}\s*(.*?)```'
    code_blocks = re.findall(pattern, markdown_text, re.DOTALL)
    return "\n".join(code_blocks)

import time
import datetime
def get_time():
    timestamp = int(time.time())  # Get the current timestamp as an integer
    # Convert the timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    
    input_string = str(dt_object)

    # Parse the input string into a datetime object
    dt_object = datetime.datetime.strptime(input_string, '%Y-%m-%d %H:%M:%S')

    # Format the datetime object to the desired string format
    formatted_string = dt_object.strftime('%Y_%m%d_%H%M%S')
    return formatted_string

def copy_and_rename_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            new_filename = filename.replace(".txt", "_chatgpt.txt")
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, new_filename)
            shutil.copy(source_file, destination_file)
            print(f"Copied and renamed {source_file} to {destination_file}")

'''
Codes Below are only for debug!
'''

if __name__ == "__main__":
    pass


