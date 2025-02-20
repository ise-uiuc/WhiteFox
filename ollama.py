import argparse
import requests
import time
import os
import json
from pathlib import Path

def clean_prompt(prompt):
    """Clean the prompt text from any markdown or other tags"""
    # Remove markdownCopy if present
    if prompt.startswith("markdownCopy"):
        prompt = prompt[len("markdownCopy"):]
    
    # Remove any userStyle tags
    prompt = prompt.replace("<userStyle>Normal</userStyle>", "")
    
    # Remove any other XML-like tags
    import re
    prompt = re.sub(r'<[^>]+>', '', prompt)
    
    # Remove any markdown tags
    if prompt.startswith("markdown"):
        prompt = prompt[len("markdown"):]
        
    # Remove extra whitespace and blank lines
    lines = [line.strip() for line in prompt.split('\n')]
    lines = [line for line in lines if line]
    prompt = '\n'.join(lines)
    
    return prompt.strip()

def process_msg(msg):
    """Process the message and extract code blocks"""
    print("Full response received:")
    print("=" * 50)
    print(msg)
    print("=" * 50)
    
    if "```" not in msg:
        print("No code block markers (```) found in response")
        return msg
        
    code_st = False 
    code = ""
    code_found = False
    
    for line in msg.splitlines():
        current_line = line.strip()
        if code_st:
            if current_line.startswith("```"):
                code_st = False
                if code_found:
                    break
                continue
            code += line + "\n"
        else:
            if current_line.startswith("```"):
                print(f"Found code block marker: '{current_line}'")
                code_st = True
                code_found = True
    
    if code:
        print("\nExtracted code:")
        print("-" * 50)
        print(code)
        print("-" * 50)
    
    return code.strip()

def generate_with_ollama(prompt, model="starcoder", temperature=1.0):
    try:
        # Clean the prompt first
        prompt = clean_prompt(prompt)
        print(f"Generating code with {model}...")
        print(f"Cleaned prompt first 200 chars:\n{prompt[:200]}")
        
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "context_length": 4096,  # Increased context length
                "top_p": 0.95,
                "repeat_penalty": 1.1,  # Added repeat penalty
            })
        
        if response.status_code == 200:
            response_json = response.json()
            result = response_json['response']
            print(f"Got response length: {len(result)}")
            return result
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error in generate_with_ollama: {str(e)}")
        return None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompt/demo")
    parser.add_argument("--outdir", type=str, default="ollama/zero-shot")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="starcoder")
    args = parser.parse_args()

    prompt_dir = Path(args.prompt_dir)
    opts = {}
    for prompt_file in prompt_dir.iterdir():
        if not prompt_file.is_file():
            continue
        opts[prompt_file.stem] = prompt_file.read_text()
        print(f"Loaded prompt from: {prompt_file}")

    outdir = args.outdir
    
    for opt_idx, opt in enumerate(opts):
        if os.path.exists(os.path.join(outdir, opt, f"{opt}_1.c")):
            print(f"Skipping {opt} - output already exists")
            continue

        print(f"\nProcessing {opt}...")
        ret = {"opt": opt}
        ret["response"] = {}
        os.makedirs(os.path.join(outdir, opt), exist_ok=True)
        
        # Save prompt
        with open(os.path.join(outdir, opt, f"prompt.txt"), "w") as f:
            f.write(opts[opt])

        # Generate code
        response = generate_with_ollama(opts[opt], args.model, args.temperature)
        if response:
            code = process_msg(response)
            if code:
                print(f"Successfully generated code for {opt}")
                
                # Save code
                with open(os.path.join(outdir, opt, f"{opt}_1.c"), "w") as f:
                    f.write(code)
                # Save full response
                with open(os.path.join(outdir, opt, f"{opt}_1.txt"), "w") as f:
                    f.write(response)
                    
                ret["response"][1] = {
                    "raw": response,
                    "code": code
                }

                # Save to outputs.json
                with open(os.path.join(outdir, "outputs.json"), "a") as f:
                    f.write(json.dumps(ret, indent=4) + "\n")
                
                print(f"Files saved in {os.path.join(outdir, opt)}")
            else:
                print("No code block found in response")
        else:
            print(f"Failed to generate code for {opt}")

        # Exit after first successful generation
        print("Generation complete, exiting...")
        break