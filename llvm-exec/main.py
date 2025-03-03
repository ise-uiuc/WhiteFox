import json
import argparse
import os

from json_parser import extract_source_from_llvm, extract_func_body_from_source
# Import functions from gen_prompt_llvm_c.py
# Note: You'll need to adjust these imports to match the actual functions in your gen_prompt_llvm_c.py
from gen_prompt_c import generate_src2nl_prompt, prepare_optimizations

def setup_directories():
    """Create all required directories"""
    directories = [
        "source-code-data/llvm/llvm-lib",
        "source-code-data/llvm/llvm-func-body",
        "source-code-data/llvm/llvm-gen-prompt",
        "source-code-data/llvm/llvm-gen-prompt/src2nl",
        "prompt/demo"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="example.json")
    parser.add_argument("--llvm-source", type=str, default="")
    parser.add_argument("--model", type=str, choices=['ollama', 'starcoder'], default='ollama')
    parser.add_argument("--template", type=str, default="template_src2nl_llvm.md")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip extraction of source files")
    args = parser.parse_args()

    print(f"Loading JSON from {args.json}")
    with open(args.json, "r") as file:
        data = json.load(file)

    print(f"Using LLVM source from: {args.llvm_source}")
    print(f"Using model: {args.model}")

    # Create necessary directories
    setup_directories()

    # Update paths with LLVM source
    print("Updating paths with LLVM source location...")
    for k in data.keys():
        for idx, _ in enumerate(data[k]["hints"]):
            if args.llvm_source:
                data[k]["hints"][idx]["codes"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][0])
                try:
                    if len(data[k]["hints"][idx]["codes"]) > 1:
                        data[k]["hints"][idx]["codes"][1] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][1])
                except:
                    pass
                
                if "examples" in data[k]["hints"][idx]:
                    data[k]["hints"][idx]["examples"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["examples"][0])
                
                if "specific_ir" in data[k]["hints"][idx] and data[k]["hints"][idx]["specific_ir"]:
                    data[k]["hints"][idx]["specific_ir"] = os.path.join(args.llvm_source, data[k]["hints"][idx]["specific_ir"])

# Then modify the extraction steps to check this flag
if not args.skip_extraction:
    print("\nStep 1: Extracting source files...")
    extract_source_from_llvm(data)

    print("\nStep 2: Extracting function bodies...")
    extract_func_body_from_source(data)

    print("\nStep 3: Generating prompts...")
    # Define output directory for src2nl prompts
    src2nl_dir = "source-code-data/llvm/llvm-gen-prompt/src2nl"
    os.makedirs(src2nl_dir, exist_ok=True)

    # Step 3a: Generate src2nl prompts using gen_prompt_llvm_c.py
    print("\nStep 3a: Generating source-to-requirements prompts...")
    
    # Convert data to format expected by gen_prompt_llvm_c.py
    optimizations = []
    for pass_name, pass_info in data.items():
        opt_info = pass_info.copy()
        opt_info["pass_name"] = pass_name
        optimizations.append(opt_info)
    
    # Generate prompts for each optimization
    for opt_info in optimizations:
        generate_src2nl_prompt(opt_info, args.template, src2nl_dir)

    print("\nDone! Generated src2nl prompts can be found in:")
    print(f"- Source files: source-code-data/llvm/llvm-lib/")
    print(f"- Function bodies: source-code-data/llvm/llvm-func-body/")
    print(f"- Generated prompts: {src2nl_dir}/")