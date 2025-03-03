#!/usr/bin/env python3
import json
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Any, List

def replace_single_braces(replacement, string, to_be_replaced="{}"):
    """Replace the first occurrence of to_be_replaced with replacement in string"""
    return string.replace(f"{to_be_replaced}", replacement, 1)

def read_llvm_pass_code(code_path: str) -> str:
    """Read the source code of an LLVM pass from the given path"""
    if not os.path.exists(code_path):
        print(f"Warning: Code path {code_path} does not exist")
        return ""
    
    try:
        with open(code_path, 'r') as f:
            content = f.read()
            print(f"Successfully read {len(content)} characters from {code_path}")
            return content
    except Exception as e:
        print(f"Error reading code from {code_path}: {e}")
        return ""

def extract_pass_name(filename):
    """Extract pass name from the filename, removing any trailing digits and .cpp extension"""
    base_name = os.path.basename(filename)
    # Remove .cpp extension
    name = os.path.splitext(base_name)[0]
    # Remove trailing digits
    name = re.sub(r'\d+$', '', name)
    return name

def insert_code_content(optimizations):
    """Insert the actual code content into the optimizations data"""
    for opt_info in optimizations:
        pass_name = opt_info.get("pass_name", "")
        print(f"Processing pass: {pass_name}")
        
        hints = opt_info.get("hints", [])
        for hint in hints:
            if "codes" in hint:
                code_contents = []
                for code_path in hint["codes"]:
                    # Try to read the code file
                    print(f"Trying to read: {code_path}")
                    content = read_llvm_pass_code(code_path)
                    if content:
                        code_contents.append(content)
                    else:
                        # If file not found, add a placeholder message
                        code_contents.append(f"// Code file not found: {code_path}")
                
                # Store the actual code content in the data
                hint["code_contents"] = "\n\n".join(code_contents)
    
    return optimizations

def extract_function_bodies(pass_code: str, func_names: List[str]) -> str:
    """
    Extract the function bodies mentioned in func_names from the pass code
    This is a simplified version and may need enhancement for complex cases
    """
    if not pass_code or not func_names:
        return pass_code
    
    results = []
    lines = pass_code.split('\n')
    in_function = False
    current_function = ""
    current_body = []
    brace_count = 0
    
    for line in lines:
        # Check if we're at the start of a function we care about
        for func_name in func_names:
            if func_name in line and '{' in line:
                in_function = True
                current_function = func_name
                current_body = [line]
                brace_count = line.count('{') - line.count('}')
                break
        
        # If we're inside a function we care about
        if in_function:  # Fixed indentation logic
            current_body.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # Check if we've reached the end of the function
            if brace_count <= 0:
                in_function = False
                if current_function:
                    results.append('\n'.join(current_body))
                current_body = []
    
    return '\n\n'.join(results)

def generate_src2nl_prompt(opt_info: Dict[str, Any], template_path: str, output_dir: str) -> None:
    """Generate a prompt asking for description of C programs that trigger the optimization"""
    pass_name = opt_info.get("pass_name", "")
    if not pass_name:
        print("Error: Pass name not specified in optimization info")
        return
    
    # Get code contents, function names, and target line
    code_contents = []
    func_names = []
    target_line = ""
    
    hints = opt_info.get("hints", [])
    for hint in hints:
        if "code_contents" in hint:
            code_contents.append(hint["code_contents"])
        if "func" in hint:
            if isinstance(hint["func"], list):
                func_names.extend(hint["func"])
            else:
                func_names.append(hint["func"])
        if "target_line" in hint:
            target_line = hint["target_line"]
    
    # If we have a func_body_file, use that directly
    if "func_body_file" in opt_info and opt_info["func_body_file"]:
        pass_code = read_llvm_pass_code(opt_info["func_body_file"])
    else:
        # Combine all code contents
        pass_code = "\n\n".join(code_contents)
        
        # Extract relevant function bodies if function names are provided
        if func_names and pass_code:
            pass_code = extract_function_bodies(pass_code, func_names)
    
    # Read the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Create the prompt by replacing placeholders
    prompt = template
    prompt = replace_single_braces(pass_code, prompt, "{source_llvm}")
    prompt = replace_single_braces(target_line, prompt, "{target_line}")
    prompt = replace_single_braces(pass_name, prompt, "{Passname}")
    if func_names and len(func_names) > 0:
        prompt = replace_single_braces(func_names[0], prompt, "{first_function}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write prompt to file
    output_file = os.path.join(output_dir, f"{pass_name}_oneshot_0.txt")
    with open(output_file, 'w') as f:
        f.write(prompt)
    
    print(f"Generated src2nl prompt for {pass_name} at {output_file}")
    
def prepare_optimizations(json_path: str) -> List[Dict[str, Any]]:
    """Parse the optimization JSON and prepare data for prompt generation"""
    with open(json_path, 'r') as f:
        opt_data = json.load(f)
    
    optimizations = []
    for pass_name, pass_info in opt_data.items():
        # Create a new dict with pass name included
        opt_info = pass_info.copy()
        opt_info["pass_name"] = pass_name
        optimizations.append(opt_info)
    
    return optimizations

def generate_from_func_bodies(func_body_dir, template_path, output_dir, json_path=None):
    """Generate prompts from all function body files in the directory"""
    # Load JSON file if provided for additional info
    pass_info = {}
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            pass_info = json.load(f)
    
    # Find all function body files
    func_body_files = []
    for root, _, files in os.walk(func_body_dir):
        for file in files:
            if file.endswith('.cpp'):
                func_body_files.append(os.path.join(root, file))
    
    print(f"Found {len(func_body_files)} function body files")
    
    # Read the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Process each function body file
    for func_body_path in func_body_files:
        # Extract pass name from filename
        pass_name = extract_pass_name(func_body_path)
        print(f"Processing {pass_name} from {func_body_path}")
        
        # Read the function body file directly
        pass_code = ""
        try:
            with open(func_body_path, 'r') as f:
                pass_code = f.read()
                print(f"Read {len(pass_code)} bytes from {func_body_path}")
        except Exception as e:
            print(f"Error reading file {func_body_path}: {e}")
            continue
        
        if not pass_code:
            print(f"Skipping {pass_name} because no code was read")
            continue
        
        # Get target line from JSON (if available)
        target_line = ""
        if pass_name in pass_info and "hints" in pass_info[pass_name] and len(pass_info[pass_name]["hints"]) > 0:
            if "target_line" in pass_info[pass_name]["hints"][0]:
                target_line = pass_info[pass_name]["hints"][0]["target_line"]
        
        # If no target line in JSON, try to find a comment with "target line" in the code
        # Get target line from JSON (if available)
        target_line = ""
        print(f"Looking for target line for {pass_name}")
        if pass_name in pass_info:
            print(f"Found {pass_name} in JSON")
            if "hints" in pass_info[pass_name] and len(pass_info[pass_name]["hints"]) > 0:
                print(f"Found hints for {pass_name}")
                if "target_line" in pass_info[pass_name]["hints"][0]:
                    target_line = pass_info[pass_name]["hints"][0]["target_line"]
                    print(f"Found target line: {target_line}")
        else:
            print(f"{pass_name} not found in JSON")
        
        # If still no target line, try to make an educated guess based on patterns
        if not target_line:
            # Look for lines with function calls, assignments, or returns
            patterns = [
                r'\w+\s*=\s*\w+\s*\+\s*\w+',  # x = y + z
                r'return\s+\w+',              # return x
                r'\w+\(\w+,\s*\w+\)',         # func(x, y)
                r'if\s*\(\w+\s*==\s*\w+\)'    # if (x == y)
            ]
            
            lines = pass_code.split("\n")
            for line in lines:
                for pattern in patterns:
                    if re.search(pattern, line):
                        target_line = line.strip()
                        print(f"Guessed target line based on pattern: {target_line}")
                        break
                if target_line:
                    break
        
        # Create prompt by replacing placeholders - use direct string replacement
        prompt = template.replace("{source_llvm}", pass_code)
        prompt = prompt.replace("{target_line}", target_line)
        prompt = prompt.replace("{Passname}", pass_name)
        
        # Save prompt
        output_file = os.path.join(output_dir, f"{pass_name}_oneshot_0.txt")
        with open(output_file, 'w') as f:
            f.write(prompt)
        
        print(f"Generated prompt at {output_file}")

def generate_src2nl_prompt(opt_info: Dict[str, Any], template_path: str, output_dir: str) -> None:
    """Generate a prompt asking for description of C programs that trigger the optimization"""
    pass_name = opt_info.get("pass_name", "")
    if not pass_name:
        print("Error: Pass name not specified in optimization info")
        return
    
    # Get code contents, function names, and target line
    code_contents = []
    func_names = []
    target_line = ""
    
    hints = opt_info.get("hints", [])
    for hint in hints:
        if "code_contents" in hint:
            code_contents.append(hint["code_contents"])
        if "func" in hint:
            if isinstance(hint["func"], list):
                func_names.extend(hint["func"])
            else:
                func_names.append(hint["func"])
        if "target_line" in hint:
            target_line = hint["target_line"]
    
    # Combine all code contents
    pass_code = "\n\n".join(code_contents)
    
    # Extract relevant function bodies if function names are provided
    if func_names and pass_code:
        pass_code = extract_function_bodies(pass_code, func_names)
    
    # Read the template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Create the prompt by replacing placeholders
    prompt = template
    prompt = replace_single_braces(pass_code, prompt, "{source_llvm}")
    prompt = replace_single_braces(target_line, prompt, "{target_line}")
    prompt = replace_single_braces(pass_name, prompt, "{Passname}")
    if func_names and len(func_names) > 0:
        prompt = replace_single_braces(func_names[0], prompt, "{first_function}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write prompt to file
    output_file = os.path.join(output_dir, f"{pass_name}_oneshot_0.txt")
    with open(output_file, 'w') as f:
        f.write(prompt)
    
    print(f"Generated src2nl prompt for {pass_name} at {output_file}")
    
def prepare_optimizations(json_path: str) -> List[Dict[str, Any]]:
    """Parse the optimization JSON and prepare data for prompt generation"""
    with open(json_path, 'r') as f:
        opt_data = json.load(f)
    
    optimizations = []
    for pass_name, pass_info in opt_data.items():
        # Create a new dict with pass name included
        opt_info = pass_info.copy()
        opt_info["pass_name"] = pass_name
        optimizations.append(opt_info)
    
    return optimizations

def main():
    parser = argparse.ArgumentParser(description="Generate src2nl prompts for LLVM C optimizations")
    parser.add_argument("--optpath", help="Path to JSON file with optimization details")
    parser.add_argument("--json", help="Alternative to --optpath: Path to JSON file with optimization details")
    parser.add_argument("--template", required=True, help="Path to template file for src2nl prompts")
    parser.add_argument("--outdir", default="source-code-data/llvm/llvm-gen-prompt/src2nl", 
                        help="Output directory for prompts")
    parser.add_argument("--llvm-source", default="", help="Path to LLVM source code to prepend to paths")
    parser.add_argument("--func-body-dir", default="source-code-data/llvm/llvm-func-body",
                        help="Directory containing function body files")
    parser.add_argument("--all-func-bodies", action="store_true", 
                        help="Generate prompts for all function body files")
    
    args = parser.parse_args()
    
    # Support both --optpath and --json
    json_path = args.optpath if args.optpath else args.json
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.all_func_bodies:
        # Generate prompts from all function body files
        generate_from_func_bodies(args.func_body_dir, args.template, args.outdir, json_path)
    elif json_path:
        # Traditional approach using JSON file
        optimizations = prepare_optimizations(json_path)
        
        # Update paths with LLVM source if provided
        if args.llvm_source:
            for opt_info in optimizations:
                for hint in opt_info.get("hints", []):
                    if "codes" in hint:
                        for i, code_path in enumerate(hint["codes"]):
                            hint["codes"][i] = os.path.join(args.llvm_source, code_path)
        
        # Insert code content
        optimizations = insert_code_content(optimizations)
        
        # Generate prompts
        for opt_info in optimizations:
            generate_src2nl_prompt(opt_info, args.template, args.outdir)
    else:
        print("Error: Must provide either --optpath/--json or --all-func-bodies")

if __name__ == "__main__":
    main()