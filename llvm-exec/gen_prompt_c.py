import os
from typing import List
from utils import create_file, remove_extra_blank_lines, remove_commented_lines
from pathlib import Path

def replace_single_braces(replacement, string, to_be_replaced="{}"):
    return string.replace(f"{to_be_replaced}", replacement, 1)

def examples_code(examples: List[str]):
    template = """# C Code begins

```c
{example}
```
# C Code ends
"""
    code = "\n".join([template.format(example=example) for example in examples])
    return code

def gen_prompt_nl2test(
    passname: str,
    index: int,
    file_string: str,
    gen_dir_path: str,
    examples: List[str],
    description: str = None
):
    """
    This function is used to generate prompts for test generation.
    Supports both regular test generation and feedback-based generation.
    """
    # Step 0: Check whether it is a feedback loop.
    is_feedback = False
    if "feedback" in file_string:
        is_feedback = True

    # Step 1: If description is not provided, try to get it from previously generated requirements
    if not description:
        src2nl_dir = os.path.join("source-code-data/llvm/llvm-gen-prompt", "src2nl")
        src2nl_file = os.path.join(src2nl_dir, f"{passname}_oneshot_{index}.txt")
        
        if os.path.exists(src2nl_file):
            with open(src2nl_file, 'r') as f:
                description = f.read()
        else:
            raise ValueError(f"No description provided and couldn't find src2nl file: {src2nl_file}")

    # Step 2: Get the appropriate template
    template_path = f"template_{file_string}.md"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
        
    with open(template_path, "r") as template:
        tpl = template.read()

    # Step 3: Fill in the template
    tpl = replace_single_braces(description, tpl, "{Description}")
    if is_feedback and examples:
        tpl = replace_single_braces(examples_code(examples), tpl, "{Examples}")

    # Step 4: Generate the output file
    gen_dir_path = gen_dir_path if gen_dir_path else "source-code-data/llvm/llvm-gen-prompt"
    gen_dir = os.path.join(gen_dir_path)
    os.makedirs(gen_dir, exist_ok=True)
    
    gen_file_name = f"{passname}_oneshot_{index}.txt"
    gen_file = os.path.join(gen_dir, gen_file_name)
    create_file(gen_file)

    with open(gen_file, "w") as file:
        file.write(tpl)

    remove_extra_blank_lines(gen_file)
    print(f"Generated prompt file: {gen_file}")
    return

def gen_src2nl_prompt_from_template(
    func_list: list,
    func_body_list: list,
    example: str,
    passname: str,
    target_line: str,
    index: int,
    file_string: str
):
    """
    Generate source-to-natural-language prompts for LLVM passes.
    Uses the function bodies and target lines to create prompts.
    """
    # Get template
    template_path = os.path.join("../template-llvm", "gpt4-src2mixnl-oneshot.txt")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
        
    with open(template_path, "r") as template:
        tpl = template.read()
    
    # Fill template
    tpl = replace_single_braces(target_line, tpl, "{target_line}")
    tpl = replace_single_braces(func_list[0], tpl, "{first_fuction}")
    tpl = replace_single_braces(passname, tpl, "{Passname}")
    
    # Handle function bodies properly
    func_body_text = "\n\n".join([body for body in func_body_list if body is not None])
    tpl = replace_single_braces(func_body_text, tpl, "{source_llvm}")

    # Generate output file
    gen_dir = os.path.join("source-code-data/llvm/llvm-gen-prompt", "src2nl")
    os.makedirs(gen_dir, exist_ok=True)
    
    gen_file = os.path.join(gen_dir, f"{passname}_oneshot_{index}.txt")
    create_file(gen_file)

    with open(gen_file, "w") as file:
        file.write(tpl)

    remove_extra_blank_lines(gen_file)
    print(f"Generated src2nl prompt file: {gen_file}")
    return

def gen_prompt_from_template(
    func_list: list,
    func_body_list: list,
    example: str,
    passname: str,
    target_line: str,
    index: int,
    file_string: str,
    gen_dir_path: str = None,
    examples: List[str] = None,
    description: str = None
) -> str:
    """
    Main entry point for prompt generation.
    Handles both src2nl and nl2test generation.
    """
    if "src2nl" in file_string:
        gen_src2nl_prompt_from_template(
            func_list,
            func_body_list,
            example,
            passname,
            target_line,
            index,
            file_string
        )
        return

    if "ollama_c_" in file_string or "starcoder_c_" in file_string:
        gen_prompt_nl2test(
            passname,
            index,
            file_string,
            gen_dir_path,
            examples,
            description
        )
        return

    raise ValueError(f"Unsupported file_string: {file_string}")

if __name__ == "__main__":
    # Test functions
    def test_src2nl():
        passname = "ADCEPass"
        index = 0
        gen_src2nl_prompt_from_template(
            ["void testFunc()"],
            ["void testFunc() { int x = 5; return; }"],
            "example code",
            passname,
            "test target line",
            index,
            "src2nl"
        )

    def test_prompt_nl2test():
        passname = "ADCEPass"
        index = 0
        file_string = "ollama_c_deadarg"
        gen_prompt_nl2test(
            passname,
            index,
            file_string,
            gen_dir_path="test",
            examples=["int main(void) { return 0; }"],
            description="Test description for regular test"
        )

    test_src2nl()
    test_prompt_nl2test()