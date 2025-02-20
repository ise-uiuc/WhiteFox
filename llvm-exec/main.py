import json
import argparse
import os

from json_parser import extract_source_from_llvm, extract_func_body_from_source, gen_prompt
from gen_prompt_c import gen_prompt_nl2test
from utils import statistics

def setup_directories():
    """Create all required directories"""
    directories = [
        "source-code-data/llvm/llvm-lib",
        "source-code-data/llvm/llvm-func-body",
        "source-code-data/llvm/llvm-gen-prompt",
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
            data[k]["hints"][idx]["codes"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][0])
            try:
                data[k]["hints"][idx]["codes"][1] = os.path.join(args.llvm_source, data[k]["hints"][idx]["codes"][1])
            except:
                pass
            data[k]["hints"][idx]["examples"][0] = os.path.join(args.llvm_source, data[k]["hints"][idx]["examples"][0])
            data[k]["hints"][idx]["specific_ir"] = os.path.join(args.llvm_source, data[k]["hints"][idx]["specific_ir"])

    print("\nStep 1: Extracting source files...")
    extract_source_from_llvm(data)

    print("\nStep 2: Extracting function bodies...")
    extract_func_body_from_source(data)

    print("\nStep 3: Generating prompts...")
    model_prefix = "ollama_c_" if args.model == 'ollama' else "starcoder_c_"
    gen_dir = "prompt/demo" if args.model == 'ollama' else "source-code-data/llvm/llvm-gen-prompt"

    # Step 3a: First generate src2nl (source → requirements)
    print("\nStep 3a: Generating source-to-requirements prompts...")
    gen_prompt(data, file_string="src2nl", gen_dir_path=gen_dir)

    # Step 3b: Then generate nl2test using those requirements
    print("\nStep 3b: Generating test prompts...")
    for passname, pass_info in data.items():
        print(f"\nProcessing {passname} for tests...")
        for index, hint in enumerate(pass_info["hints"]):
            try:
                # Generate without feedback
                description = f"Generate C code that demonstrates the {passname} optimization. "
                description += f"Focus on the pattern: {hint['target_line']}"
                
                print(f"  Generating deadarg prompt for {passname} index {index}")
                gen_prompt_nl2test(
                    passname,
                    index,
                    model_prefix + "deadarg",
                    gen_dir,
                    None,
                    description=description
                )
                
                # Generate with feedback
                print(f"  Generating feedback prompt for {passname} index {index}")
                gen_prompt_nl2test(
                    passname,
                    index,
                    model_prefix + "feedback",
                    gen_dir,
                    ["int main(void) { return 0; }", "int main(void) { return 1; }"],
                    description=description
                )
            except Exception as e:
                print(f"Error generating test prompts for {passname} index {index}: {str(e)}")
                raise

    print("\nDone! Generated files can be found in:")
    print(f"- Source files: source-code-data/llvm/llvm-lib/")
    print(f"- Function bodies: source-code-data/llvm/llvm-func-body/")
    print(f"- Generated prompts: {gen_dir}/")