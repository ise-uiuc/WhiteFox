"""
Ollama integration for WhiteFox code generation.

Usage:
  python ollama_gen.py --prompt-dir=prompts --output-dir=ollama-generated \
  --model=codellama:7b --n=10 --temperature=1.0
"""

import argparse
import time
import json
from pathlib import Path
import logging
from pprint import pprint
import requests


EOF_STRINGS = [
    "<|endoftext|>",
    "###",
    "__output__ =",
    "if __name__",
    '"""',
    "'''",
    "# Model ends",
    "# LLVM IR ends",
    "# C++ Code ends",
]


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="codellama:7b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
    def generate(self, prompt, temperature=1.0, max_tokens=4096, n=1):
        """Generate text using Ollama API"""
        outputs = []
        
        for _ in range(n):
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
                
                generated_text = result.get("response", "")
                # Process text to stop at EOF strings
                processed_text = self._process_output(generated_text)
                outputs.append(processed_text)
                
            except Exception as e:
                logging.error(f"Error generating with Ollama: {e}")
                outputs.append("")
                
        return outputs
    
    def _process_output(self, text):
        """Process output to remove content after EOF strings"""
        min_index = len(text)
        for eof in EOF_STRINGS:
            if eof in text:
                min_index = min(min_index, text.index(eof))
        return text[:min_index]
    
    def check_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            if self.model not in available_models:
                logging.warning(f"Model {self.model} not found. Available models: {available_models}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return False


def process_msg(msg):
    """Extract code blocks from message."""
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
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-dir", type=str, default="prompts")
    parser.add_argument("--output-dir", type=str, default="ollama-generated")
    parser.add_argument("--model", type=str, default="codellama:7b",
                        help="Ollama model to use (e.g., codellama:7b, mistral:7b, llama2:7b)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434",
                        help="Ollama server base URL")
    parser.add_argument("-n", "--num", type=int, default=10,
                        help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--log-file", type=str, default="ollama-gen.log")
    parser.add_argument("--target", type=str, default="PyTorch",
                        help="Target framework (for system message)")
    
    args = parser.parse_args()
    pprint(args)
    
    logging.basicConfig(level=logging.INFO, filename=args.log_file)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    client = OllamaClient(base_url=args.base_url, model=args.model)
    
    if not client.check_connection():
        print(f"Failed to connect to Ollama server at {args.base_url}")
        print(f"Make sure Ollama is running and model {args.model} is available")
        print("You can install a model with: ollama pull <model_name>")
        return
    
    print(f"Connected to Ollama server. Using model: {args.model}")

    prompts = []
    filenames = []
    for prompt_file in prompt_dir.glob("*.txt"):
        with open(prompt_file) as f:
            content = f.read()
            system_message = f"You are a source code analyzer for {args.target}."
            full_prompt = f"System: {system_message}\n\nUser: {content}"
            prompts.append(full_prompt)
        filenames.append(prompt_file.stem)
    
    logging.info(f"Number of prompts: {len(prompts)}")
    print(f"Number of prompts: {len(prompts)}")
    
    for i, (prompt, filename) in enumerate(zip(prompts, filenames)):
        print(f"Processing {i+1}/{len(prompts)}: {filename}")
        
        output_file_dir = output_dir / filename
        output_file_dir.mkdir(exist_ok=True, parents=True)
        
        (output_file_dir / "prompt.txt").write_text(prompt)
        
        if (output_file_dir / f"{filename}_1.py").exists():
            print(f"Skipping {filename} - already exists")
            continue
        
        st_time = time.time()
        try:
            outputs = client.generate(
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n=args.num
            )
            
            for j, output in enumerate(outputs, 1):
                code = process_msg(output)
                (output_file_dir / f"{filename}_{j}.py").write_text(code)
                (output_file_dir / f"{filename}_{j}.txt").write_text(output)
            
            used_time = time.time() - st_time
            logging.info(f"Generated {len(outputs)} outputs for {filename} in {used_time:.2f}s")
            print(f"Generated {len(outputs)} outputs in {used_time:.2f}s")
            
            (output_file_dir / "timing.log").write_text(str(used_time))
            
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
