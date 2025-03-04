#!/usr/bin/env python3
"""
Script to generate C program templates from LLVM src2nl files using Ollama.
"""

import argparse
import requests
import time
import os
import re
from pathlib import Path

# Default templates for different optimization types
DEFAULT_TEMPLATES = {
    "default": """// Default template for triggering optimization
int function_with_optimizable_code(int x) {
    // Code that can be optimized
    int unused = x * 2;  // This computation might be eliminated
    return x + 5;        // Only this affects the output
}

int main() {
    int x = 10;
    int result = function_with_optimizable_code(x);
    return result;
}""",
    
    # Dead code elimination passes
    "dce": """// Template for dead code elimination
int function_with_dead_code(int x) {
    int unused = x * 2;  // This computation is never used
    return x + 5;        // Only this affects the output
}

int main() {
    int x = 10;
    int result = function_with_dead_code(x);
    return result;
}""",
    
    # Aggressive DCE and related passes
    "adce": """// Template for aggressive dead code elimination
int function_with_dead_code(int x) {
    int unused1 = x * 2;  // This computation is never used
    int unused2 = x * 3;  // This computation is never used
    
    if (x > 0) {
        int unused3 = x * 4;  // Dead code inside conditional
    }
    
    return x + 5;  // Only this affects the output
}

int main() {
    int x = 10;
    int result = function_with_dead_code(x);
    return result;
}""",

    # Dead argument elimination
    "deadarg": """// Template for dead argument elimination
static int function_with_dead_args(int x, int y, int z) {
    // Parameter z is never used in the function body
    return x + y;  // Only x and y are used
}

int main() {
    int a = 10;
    int b = 20;
    int c = 30;  // This value is passed but never used in the function
    int result = function_with_dead_args(a, b, c);
    return result;
}""",

    # Global optimizations
    "global": """// Template for global optimizations
static int global_var = 10;  // Global variable that might be optimized

static int function_using_global() {
    return global_var + 5;  // Uses the global variable
}

int main() {
    int result = function_using_global();
    return result;
}""",

    # Constant-related optimizations
    "const": """// Template for constant-related optimizations
int function_with_constants(int x) {
    const int CONSTANT_VALUE = 10;
    int result = x * CONSTANT_VALUE;  // Uses a constant value
    return result;
}

int main() {
    int x = 5;
    int result = function_with_constants(x);
    return result;
}""",

    # Loop-related optimizations
    "loop": """// Template for loop optimizations
int function_with_loop(int n) {
    int sum = 0;
    
    // Loop that could be optimized
    for (int i = 0; i < n; i++) {
        sum += i;  // Computation inside loop
    }
    
    return sum;
}

int main() {
    int n = 10;
    int result = function_with_loop(n);
    return result;
}""",

    # CFG simplification
    "cfg": """// Template for CFG simplification
int function_with_branches(int condition) {
    int result = 0;
    
    // Control flow that could be simplified
    if (condition) {
        result = 10;
    } else {
        result = 10;  // Same value as the true branch
    }
    
    return result;
}

int main() {
    int x = 10;
    int result = function_with_branches(x > 5);
    return result;
}""",

    # Instruction combining
    "instcombine": """// Template for instruction combining
int function_with_combinable_instructions(int x) {
    // Instructions that could be combined
    int temp = x + 5;
    int result = temp * 2;  // Could be combined with the previous instruction
    
    return result;
}

int main() {
    int x = 10;
    int result = function_with_combinable_instructions(x);
    return result;
}""",

    # Memory to register promotion
    "promote": """// Template for memory to register promotion
int function_with_stack_vars() {
    int array[1];  // Small array that could be promoted to a register
    array[0] = 10;
    
    return array[0];
}

int main() {
    int result = function_with_stack_vars();
    return result;
}""",

    # CSE - Common Subexpression Elimination
    "cse": """// Template for common subexpression elimination
int function_with_common_expressions(int x, int y) {
    // Expressions that are computed multiple times
    int a = x + y;
    int b = x + y;  // Same computation as above
    
    return a * b;
}

int main() {
    int x = 10;
    int y = 20;
    int result = function_with_common_expressions(x, y);
    return result;
}""",

    # GVN - Global Value Numbering
    "gvn": """// Template for global value numbering
int function_with_redundant_expressions(int x) {
    int result = 0;
    
    // Redundant expressions
    if (x > 0) {
        result = x * 2;
    } else {
        result = x * 2;  // Same computation in both branches
    }
    
    return result;
}

int main() {
    int x = 10;
    int result = function_with_redundant_expressions(x);
    return result;
}""",

    # Reassociate
    "reassociate": """// Template for reassociation
int function_with_reassociable_ops(int a, int b, int c) {
    // Operations that could be reassociated for better optimization
    return a + (b + c);  // Could be reassociated as (a + b) + c
}

int main() {
    int x = 10;
    int y = 20;
    int z = 30;
    int result = function_with_reassociable_ops(x, y, z);
    return result;
}""",

    # Loop unswitch
    "unswitch": """// Template for loop unswitching
int function_with_invariant_condition(int x, int n) {
    int sum = 0;
    
    // Loop with invariant condition
    for (int i = 0; i < n; i++) {
        if (x > 0) {  // Condition invariant to the loop
            sum += i;
        } else {
            sum += i * 2;
        }
    }
    
    return sum;
}

int main() {
    int x = 10;
    int n = 5;
    int result = function_with_invariant_condition(x, n);
    return result;
}""",

    # Jump threading
    "jump": """// Template for jump threading
int function_with_threadable_jumps(int x) {
    int result = 0;
    
    // Control flow that could be threaded
    if (x > 0) {
        if (x > 5) {
            result = 10;
        } else {
            result = 20;
        }
    } else {
        result = 30;
    }
    
    return result;
}

int main() {
    int x = 10;
    int result = function_with_threadable_jumps(x);
    return result;
}"""
}

# Standard descriptions for different optimization types
DEFAULT_DESCRIPTIONS = {
    "default": "the program contains code patterns that can be optimized by the compiler",
    
    "dce": "the program contains dead code - statements or expressions that are executed but have no effect on the output, result, or state of the program",
    
    "adce": "the program contains code that doesn't affect the program's observable behavior and can be eliminated without changing the program's semantics",
    
    "deadarg": "a function has parameters that are never used within the function body",
    
    "global": "there are global variables or functions that can be optimized based on their usage patterns",
    
    "const": "constant values are used in expressions that can be evaluated at compile time",
    
    "loop": "there are loops with patterns that can be optimized, such as invariant expressions or simplifiable control flow",
    
    "cfg": "the control flow graph can be simplified without changing program behavior, such as removing redundant branches or merging identical code paths",
    
    "instcombine": "multiple instructions can be combined into fewer, more efficient instructions",
    
    "promote": "stack-allocated variables can be promoted to register values, eliminating memory operations",
    
    "cse": "the same expression is computed multiple times and can be computed once and reused",
    
    "gvn": "redundant computations exist across different paths in the code",
    
    "reassociate": "arithmetic operations can be reassociated to enable better constant folding or other optimizations",
    
    "unswitch": "a loop contains a condition that is invariant (doesn't change) through loop iterations",
    
    "jump": "control flow can be threaded through conditional branches to eliminate unnecessary jumps"
}

def query_ollama(prompt, model="starcoder", system=None, max_retries=3):
    """Query Ollama with a prompt and retry on failure"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    if system:
        data["system"] = system
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  Retry attempt {attempt+1}/{max_retries}...")
                time.sleep(5)
                
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"  Error on attempt {attempt+1}: {e}")
    
    print(f"  Failed after {max_retries} attempts")
    return None

def extract_info_from_src2nl(src2nl_content):
    """Extract pass code, target line, and pass name from src2nl content"""
    pass_code = ""
    target_line = ""
    pass_name = ""
    
    if "# Code of the pass" in src2nl_content and "# Target line to focus on" in src2nl_content:
        code_start = src2nl_content.find("# Code of the pass") + len("# Code of the pass")
        code_end = src2nl_content.find("# Target line to focus on")
        pass_code = src2nl_content[code_start:code_end].strip()
        
        target_start = src2nl_content.find("# Target line to focus on") + len("# Target line to focus on")
        target_end = src2nl_content.find("# Pass name")
        target_line = src2nl_content[target_start:target_end].strip()
        
        pass_name_start = src2nl_content.find("# Pass name") + len("# Pass name")
        pass_name_end = src2nl_content.find("# Description") if "# Description" in src2nl_content else len(src2nl_content)
        pass_name = src2nl_content[pass_name_start:pass_name_end].strip()
    
    return pass_code, target_line, pass_name

def create_ollama_prompt(pass_code, target_line, pass_name):
    """Create a prompt for Ollama with clear examples and instructions"""
    # Build the prompt in pieces to avoid issues with nested triple quotes
    prompt = f"Based on this LLVM {pass_name} optimization pass code, generate a C program template that would trigger this optimization.\n\n"
    prompt += f"This optimization is triggered by code at this line:\n{target_line}\n\n"
    prompt += "In the pass code:\n```\n"
    prompt += f"{pass_code}\n"
    prompt += "```\n\n"
    
    prompt += "Write a C program template that would trigger this optimization. Focus only on the structure/pattern needed, using placeholders like EXPRESSION, VALUE, etc.\n\n"
    prompt += "DO NOT USE PLACEHOLDERS in your response like [Key point 1] or [Program goes here].\n\n"
    prompt += "Here are two examples of good responses:\n\n"
    
    # Example 1
    prompt += "EXAMPLE 1:\n```c\n"
    prompt += "// Template for dead code elimination\n"
    prompt += "int function_with_dead_code(int x) {\n"
    prompt += "    int unused = x * 2;  // This computation is never used\n"
    prompt += "    return x + 5;        // Only this affects the output\n"
    prompt += "}\n\n"
    prompt += "int main() {\n"
    prompt += "    int x = 10;\n"
    prompt += "    int result = function_with_dead_code(x);\n"
    prompt += "    return result;\n"
    prompt += "}\n"
    prompt += "```\n\n"
    
    # Example 2
    prompt += "EXAMPLE 2:\n```c\n"
    prompt += "// Template for constant folding\n"
    prompt += "int function_with_constants() {\n"
    prompt += "    const int VALUE1 = 10;\n"
    prompt += "    const int VALUE2 = 20;\n"
    prompt += "    return VALUE1 + VALUE2;  // Can be constant-folded\n"
    prompt += "}\n\n"
    prompt += "int main() {\n"
    prompt += "    int result = function_with_constants();\n"
    prompt += "    return result;\n"
    prompt += "}\n"
    prompt += "```\n\n"
    
    prompt += "Respond with ONLY your C program template in a code block, and a one-sentence description of what triggers this optimization."
    
    return prompt

def select_fallback_template(pass_name):
    """Select an appropriate fallback template based on the pass name"""
    pass_name_lower = pass_name.lower()
    
    # Check for matching templates
    for key in DEFAULT_TEMPLATES:
        if key in pass_name_lower:
            return DEFAULT_TEMPLATES[key], DEFAULT_DESCRIPTIONS.get(key, DEFAULT_DESCRIPTIONS["default"])
    
    # If no match found, return default
    return DEFAULT_TEMPLATES["default"], DEFAULT_DESCRIPTIONS["default"]

def is_placeholder_response(response):
    """Check if the response contains placeholders or is incomplete"""
    placeholder_patterns = [
        r"\[key point \d+\]",
        r"\[.*goes here\]",
        r"// input program",
        r"// output goes here",
        r"\[your .*\]"
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, response.lower()):
            return True
    
    # Check if it has actual code
    if "```c" not in response.lower() and "```" not in response:
        return True
        
    return False

def extract_from_response(response):
    """Extract C template and description from Ollama's response"""
    c_template = ""
    description = ""
    
    # Extract C template from code block
    code_matches = re.findall(r"```(?:c)?\n(.*?)```", response, re.DOTALL)
    if code_matches:
        c_template = code_matches[0].strip()
    
    # Extract description - everything that's not in a code block
    description_text = re.sub(r"```(?:c)?\n.*?```", "", response, flags=re.DOTALL).strip()
    if description_text:
        # Get the last sentence which often contains the summary
        sentences = re.split(r'[.!?]\s+', description_text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        if sentences:
            description = sentences[-1]
    
    return c_template, description

def create_final_output(c_template, description, pass_name):
    """Create the final requirements output"""
    if not c_template or not description:
        c_template, description = select_fallback_template(pass_name)
    
    return f"""### Please generate a valid C program that meets the requirements below. The program should contain a `main` function that returns an integer value. Please initialize all variables you define with a value. Please do not include any undefined behavior in your code. The code you generate will be used to test the correctness of the optimization.

# Description of requirements

The C program should contain the following pattern:

```c
{c_template}
```

This pattern characterizes scenarios where {description}

# C Program
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src2nl-dir", type=str, default="source-code-data/llvm/llvm-gen-prompt/src2nl",
                        help="Directory containing src2nl files")
    parser.add_argument("--output-dir", type=str, default="source-code-data/llvm/llvm-gen-prompt/requirements",
                        help="Output directory for requirements files")
    parser.add_argument("--model", type=str, default="starcoder",
                        help="Ollama model to use")
    parser.add_argument("--filter", type=str, default="",
                        help="Filter to only process files containing this string")
    parser.add_argument("--system-message", type=str, 
                        default="You are a compiler expert specializing in LLVM optimizations. Provide concise C code examples.",
                        help="System message for Ollama")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Skip Ollama and just use fallback templates")
    
    args = parser.parse_args()
    
    src2nl_dir = Path(args.src2nl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all src2nl files
    src2nl_files = list(src2nl_dir.glob("*_oneshot_*.txt"))
    if args.filter:
        src2nl_files = [f for f in src2nl_files if args.filter in f.name]
    
    print(f"Found {len(src2nl_files)} src2nl files")
    
    # Process each file
    for i, src2nl_file in enumerate(src2nl_files):
        pass_name = src2nl_file.stem.split("_oneshot_")[0]
        print(f"[{i+1}/{len(src2nl_files)}] Processing {pass_name}")
        
        # Read src2nl file
        with open(src2nl_file, 'r') as f:
            src2nl_content = f.read()
        
        # Extract information
        pass_code, target_line, file_pass_name = extract_info_from_src2nl(src2nl_content)
        if file_pass_name:
            pass_name = file_pass_name
        
        c_template = ""
        description = ""
        
        if not args.skip_ollama:
            # Create prompt for Ollama
            ollama_prompt = create_ollama_prompt(pass_code, target_line, pass_name)
            
            # Query Ollama
            print(f"  Querying Ollama for {pass_name}...")
            response = query_ollama(ollama_prompt, args.model, args.system_message)
            
            if response and not is_placeholder_response(response):
                print(f"  Got valid response")
                # Extract template and description
                c_template, description = extract_from_response(response)
                
                # Save raw response
                raw_file = output_dir / f"{pass_name}_ollama_response.txt"
                with open(raw_file, 'w') as f:
                    f.write(response)
            else:
                print(f"  Got placeholder response or no response, using fallback")
        else:
            print(f"  Skipping Ollama, using fallback template")
        
        # If we didn't get usable content, use fallback
        if not c_template or not description:
            c_template, description = select_fallback_template(pass_name)
        
        # Create final output
        final_output = create_final_output(c_template, description, pass_name)
        
        # Save to file
        output_file = output_dir / f"{pass_name}_requirements.txt"
        with open(output_file, 'w') as f:
            f.write(final_output)
        
        print(f"  Saved requirements to {output_file}")
        
        # Pause between requests
        time.sleep(2)