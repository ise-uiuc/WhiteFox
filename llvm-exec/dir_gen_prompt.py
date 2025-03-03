#!/usr/bin/env python3
import os
import re
import argparse

def extract_pass_name(filename):
    """Extract pass name from the filename, removing any trailing digits and .cpp extension"""
    base_name = os.path.basename(filename)
    # Remove .cpp extension
    name = os.path.splitext(base_name)[0]
    # Remove trailing digits
    name = re.sub(r'\d+$', '', name)
    return name

def main():
    parser = argparse.ArgumentParser(description="Generate src2nl prompts for LLVM optimizations")
    parser.add_argument("--template", default="template_src2nl_llvm.md", help="Template file path")
    parser.add_argument("--outdir", default="source-code-data/llvm/llvm-gen-prompt/src2nl", help="Output directory")
    parser.add_argument("--func-body-dir", default="source-code-data/llvm/llvm-func-body", help="Function body directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load template
    with open(args.template, 'r') as f:
        template = f.read()
    
    # Get all cpp files
    func_body_files = []
    for file in os.listdir(args.func_body_dir):
        if file.endswith('.cpp'):
            func_body_files.append(os.path.join(args.func_body_dir, file))
    
    print(f"Found {len(func_body_files)} function body files")
    
    # Process each function body file
    for func_body_path in func_body_files:
        # Extract pass name
        pass_name = extract_pass_name(func_body_path)
        print(f"Processing {pass_name}")
        
        # Read the function body
        with open(func_body_path, 'r') as f:
            pass_code = f.read()
            print(f"Read {len(pass_code)} bytes")
        
        # Find a target line (simple heuristic)
        target_line = ""
        lines = pass_code.split('\n')
        for line in lines:
            if re.search(r'\w+\s*=\s*\w+\s*\+\s*\w+', line):  # x = y + z pattern
                target_line = line.strip()
                break
            elif "target line" in line.lower():
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    target_line = lines[idx + 1].strip()
                    break
        
        # Create the prompt
        prompt = template.replace("{source_llvm}", pass_code)
        prompt = prompt.replace("{target_line}", target_line)
        prompt = prompt.replace("{Passname}", pass_name)
        
        # Save the prompt
        output_file = os.path.join(args.outdir, f"{pass_name}_oneshot_0.txt")
        with open(output_file, 'w') as f:
            f.write(prompt)
        
        print(f"Generated prompt at {output_file}")

if __name__ == "__main__":
    main()