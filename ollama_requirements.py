#!/usr/bin/env python3
"""
Ollama-based requirements generator for WhiteFox.
Alternative to gpt4.py that uses Ollama instead of OpenAI.

Usage:
python ollama_requirements.py --prompt-dir=MyPrompts/torch-inductor/req2test \
    --outdir=MyRequirements \
    --model=codellama:7b \
    --temperature=0.0 \
    --target="PyTorch"
"""

import argparse
import time
import json
from pathlib import Path
import requests
import logging

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="codellama:7b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
    def generate(self, prompt, temperature=0.0, max_tokens=512):
        """Generate text using Ollama API"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logging.error(f"Error generating with Ollama: {e}")
            raise e
    
    def check_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            if self.model not in available_models:
                print(f"Model {self.model} not found. Available models: {available_models}")
                print(f"You can install the model with: ollama pull {self.model}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")
            return False


def process_msg(msg):
    """Extract code blocks or return the message as-is."""
    if "```" not in msg:
        return msg
    
    code_st = False
    code = ""
    for line in msg.splitlines():
        if code_st:
            if line.strip().startswith("```"):
                code_st = False
                continue
            code += line + "\n"
        else:
            if line.strip().startswith("```"):
                code_st = True
    return code.strip() if code else msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, default="codellama:7b")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target", type=str, default="PyTorch")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of generations per prompt")
    
    args = parser.parse_args()
    
    client = OllamaClient(base_url=args.base_url, model=args.model)
    
    if not client.check_connection():
        return 1
    
    print(f"Using Ollama model: {args.model}")
    
    system_message = f"You are a source code analyzer for {args.target}."
    
    prompt_dir = Path(args.prompt_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    prompts = {}
    for prompt_file in prompt_dir.iterdir():
        if not prompt_file.is_file() or prompt_file.suffix != '.txt':
            continue
        prompts[prompt_file.stem] = prompt_file.read_text().strip()
    
    print(f"Found {len(prompts)} prompts")
    
    for opt_idx, (opt_name, user_input) in enumerate(prompts.items()):
        opt_outdir = outdir / opt_name
        opt_outdir.mkdir(exist_ok=True)
        
        (opt_outdir / "prompt.txt").write_text(user_input)
        
        if (opt_outdir / f"{opt_name}_1.txt").exists():
            print(f"Skipping {opt_name} - already exists")
            continue
        
        print(f"[{opt_idx+1}/{len(prompts)}] Processing {opt_name}")
        
        full_prompt = f"System: {system_message}\n\nUser: {user_input}\n\nAssistant:"
        
        for attempt in range(3):
            try:
                t_start = time.time()
                response = client.generate(
                    full_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
                g_time = time.time() - t_start
                print(f"Generated in {g_time:.2f}s")
                
                processed_response = process_msg(response)
                
                (opt_outdir / f"{opt_name}_1.txt").write_text(processed_response)
                (opt_outdir / f"{opt_name}_1_raw.txt").write_text(response)

                metadata = {
                    "model": args.model,
                    "temperature": args.temperature,
                    "generation_time": g_time,
                    "prompt_length": len(full_prompt),
                    "response_length": len(response)
                }
                (opt_outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                
                break
                
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    print(f"Failed to process {opt_name} after 3 attempts")
                else:
                    time.sleep(5)
    
    print("Requirements generation completed!")
    return 0


if __name__ == "__main__":
    exit(main())
