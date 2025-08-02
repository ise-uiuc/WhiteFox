"""
Ollama service backend for WhiteFox.

Usage:
```
python ollama_service.py --prompt-dir=/path/to/prompts --output-dir=/path/to/output \
    --model=codellama:7b --num=10 --sleep-time=30
```

The service scans the prompt directory every 30 seconds and generates code for new prompts.
"""

import argparse
import time
import json
from pathlib import Path
import logging
from datetime import datetime
import requests
import os


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


class Logger:
    def __init__(self, log_file: Path, is_print=False) -> None:
        self.log_file = log_file
        self.is_print = is_print

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        with open(self.log_file, "a") as f:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            f.write("====================\n")
            f.write(f"[{formatted_datetime}] Start Ollama service logging.\n")

    def log(self, msg):
        if self.is_print:
            print(msg)
        timestamp = datetime.now().strftime("%d.%b %Y %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")


class OllamaService:
    def __init__(self, base_url="http://localhost:11434", model="codellama:7b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
    def generate(self, prompt, temperature=1.0, max_tokens=4096):
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
            
            generated_text = result.get("response", "")
            # Process text to stop at EOF strings
            return self._process_output(generated_text)
            
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")
    
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
                return False, f"Model {self.model} not found. Available: {available_models}"
                
            return True, "Connection successful"
            
        except Exception as e:
            return False, f"Failed to connect to Ollama: {e}"


def scan_prompt(prompt_dirs: Path, existing_prompts: set, target: str = None):
    """Scan for new prompts to process"""
    new_prompts = set()
    for target_dir in prompt_dirs.iterdir():
        if not target_dir.is_dir():
            continue
        if target is not None:
            if target_dir.name != target:
                continue
        for prompt_dir in target_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            for prompt_file in prompt_dir.iterdir():
                if not prompt_file.is_file():
                    continue
                if prompt_file.suffix != ".txt":
                    continue
                if prompt_file in existing_prompts:
                    continue
                new_prompts.add(prompt_file)
    return new_prompts


def clean_code(msg: str) -> str:
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
    parser.add_argument("--prompt-dir", type=str, required=True,
                        help="Directory containing prompts to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save generated outputs")
    parser.add_argument("--model", type=str, default="codellama:7b",
                        help="Ollama model to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434",
                        help="Ollama server base URL")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--sleep-time", type=int, default=30,
                        help="Sleep time between scans (seconds)")
    parser.add_argument("--target", type=str, default=None,
                        help="Specific target to process")
    parser.add_argument("--log-file", type=str, default="ollama-service.log")
    
    args = parser.parse_args()
    
    logger = Logger(Path(args.log_file), is_print=True)
    
    ollama = OllamaService(base_url=args.base_url, model=args.model)
    
    success, message = ollama.check_connection()
    if not success:
        logger.log(f"Failed to connect to Ollama: {message}")
        logger.log("Make sure Ollama is running and the model is available")
        logger.log(f"You can install a model with: ollama pull {args.model}")
        return
    
    logger.log(f"Connected to Ollama server. Using model: {args.model}")
    
    prompt_dirs = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    existing_prompts = set()
    
    logger.log("Starting Ollama service...")
    
    while True:
        try:
            new_prompts = scan_prompt(prompt_dirs, existing_prompts, args.target)
            
            if not new_prompts:
                logger.log(f"No new prompts found. Sleeping for {args.sleep_time}s...")
                time.sleep(args.sleep_time)
                continue
            
            logger.log(f"Found {len(new_prompts)} new prompts to process")
            
            for prompt_file in new_prompts:
                try:
                    prompt_content = prompt_file.read_text()
                    
                    relative_path = prompt_file.relative_to(prompt_dirs)
                    target_name = relative_path.parts[0]
                    step_name = relative_path.parts[1]
                    prompt_name = prompt_file.stem
                    
                    output_path = output_dir / target_name / step_name / prompt_name
                    output_path.mkdir(exist_ok=True, parents=True)
                    
                    if (output_path / f"{prompt_name}_1.py").exists():
                        logger.log(f"Skipping {prompt_file} - already processed")
                        existing_prompts.add(prompt_file)
                        continue
                    
                    logger.log(f"Processing {prompt_file}")
                    
                    (output_path / "prompt.txt").write_text(prompt_content)
                    
                    for i in range(1, args.num + 1):
                        st_time = time.time()
                        
                        try:
                            generated_text = ollama.generate(
                                prompt_content,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens
                            )
                            
                            code = clean_code(generated_text)
                            (output_path / f"{prompt_name}_{i}.py").write_text(code)
                            (output_path / f"{prompt_name}_{i}.txt").write_text(generated_text)
                            
                            used_time = time.time() - st_time
                            logger.log(f"Generated {prompt_name}_{i} in {used_time:.2f}s")
                            
                        except Exception as e:
                            logger.log(f"Error generating {prompt_name}_{i}: {e}")
                            (output_path / f"{prompt_name}_{i}.py").write_text(f"# Error: {e}")
                    
                    existing_prompts.add(prompt_file)
                    logger.log(f"Completed processing {prompt_file}")
                    
                except Exception as e:
                    logger.log(f"Error processing {prompt_file}: {e}")
                    existing_prompts.add(prompt_file)  
            
        except KeyboardInterrupt:
            logger.log("Service stopped by user")
            break
        except Exception as e:
            logger.log(f"Unexpected error: {e}")
            time.sleep(args.sleep_time)


if __name__ == "__main__":
    main()
