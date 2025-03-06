#!/usr/bin/env python3
"""
Script to read requirements from llvm-exec/source-code-data/llvm/llvm-exec/requirements,
generate C programs using Ollama's StarCoder model, and save the outputs to a specified directory.

Usage:
    python starcoder_ollama.py --requirements-dir=/path/to/llvm-exec/source-code-data/llvm/llvm-exec/requirements --output-dir=/path/to/output --num=10

The script will:
1. Scan the requirements directory for .txt files
2. Send each requirement to the Ollama StarCoder model
3. Extract C programs from the responses
4. Save the extracted C programs directly to the output directory with passname_timestamp naming
"""

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
import requests


class Logger:
    def __init__(self, log_file: Path, is_print=True) -> None:
        self.log_file = log_file
        self.is_print = is_print

        # Initialize log file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        with open(self.log_file, "a") as f:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            f.write("====================\n")
            f.write(f"[{formatted_datetime}] Start logging.\n")

    def log(self, msg):
        if self.is_print:
            print(msg)
        timestamp = datetime.now().strftime("%d.%b %Y %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")


class OllamaStarCoder:
    def __init__(self, api_url="http://localhost:11434/api/generate", temperature=0.7) -> None:
        self.api_url = api_url
        self.temperature = temperature
        self.model = "starcoder"  # Assuming the model is loaded in Ollama

    def generate(self, prompt, num_samples=1):
        """Generate code using Ollama's StarCoder model"""
        outputs = []
        
        for _ in range(num_samples):
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
            
            try:
                response = requests.post(self.api_url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    outputs.append(result.get("response", ""))
                else:
                    print(f"Error from Ollama API: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Exception when calling Ollama API: {str(e)}")
                
        return outputs


def extract_c_program(response):
    """Extract C program from the response"""
    # First try to extract code between C code blocks
    c_code_pattern = re.compile(r"```c\n(.*?)```", re.DOTALL)
    matches = c_code_pattern.findall(response)
    
    if matches:
        return matches[0].strip()
    
    # If no C code blocks found, check for generic code blocks
    code_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = code_pattern.findall(response)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks at all, return the whole response
    return response.strip()


def scan_requirements(requirements_dir: Path, existing_reqs: set):
    """Scan for new requirement files"""
    new_reqs = set()
    
    for req_file in requirements_dir.glob("**/*.txt"):
        if req_file.is_file() and req_file not in existing_reqs:
            new_reqs.add(req_file)
            
    return new_reqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama StarCoder for LLVM requirements")
    parser.add_argument("--requirements-dir", type=str, 
                        default="llvm-exec/source-code-data/llvm/llvm-exec/requirements",
                        help="Directory containing requirement files")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to store generated code")
    parser.add_argument("--num", type=int, default=5,
                        help="Number of samples to generate per requirement")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--api-url", type=str, default="http://localhost:11434/api/generate",
                        help="Ollama API URL")
    parser.add_argument("--sleep-time", type=int, default=30,
                        help="Sleep time between scans (seconds)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run in continuous mode, checking for new requirements")
    
    args = parser.parse_args()
    
    # Set up directories
    requirements_dir = Path(args.requirements_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(output_dir / "log.txt")
    
    # Log arguments
    logger.log("Arguments for Ollama StarCoder service")
    for k, v in vars(args).items():
        logger.log(f"  {k}: {v}")
    
    # Initialize Ollama client
    ollama = OllamaStarCoder(api_url=args.api_url, temperature=args.temperature)
    
    # Track processed requirements
    existing_reqs = set()
    
    # Main loop
    while True:
        new_reqs = scan_requirements(requirements_dir, existing_reqs)
        
        if len(new_reqs) == 0:
            if not args.continuous:
                logger.log("No requirements found. Exiting.")
                break
                
            logger.log(f"No new requirements, sleeping for {args.sleep_time}s...")
            time.sleep(args.sleep_time)
            continue
        
        # Process new requirements
        length = len(new_reqs)
        logger.log(f"Found {length} new requirements, starting generation...")
        
        for idx, req_file in enumerate(new_reqs):
            existing_reqs.add(req_file)
            
            # Get the passname (requirement name)
            passname = req_file.stem
            
            logger.log(f"[{idx+1}/{length}] {passname}: generating")
            
            # Read requirement
            requirement = req_file.read_text(encoding="utf-8", errors="ignore")
            
            # Generate responses
            try:
                t_start = time.time()
                responses = ollama.generate(requirement, num_samples=args.num)
                generation_time = time.time() - t_start
                logger.log(f"[{idx+1}/{length}] {passname}: generated {len(responses)} responses in {generation_time:.2f}s")
                
                # Process each response
                for i, response in enumerate(responses):
                    # Extract C program
                    c_program = extract_c_program(response)
                    
                    # Generate timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
                    
                    # Save to file directly in output directory
                    output_file = output_dir / f"{passname}_{timestamp}.c"
                    output_file.write_text(c_program)
                    logger.log(f"Saved {output_file}")
                    
            except Exception as e:
                logger.log(f"Error processing {passname}: {str(e)}")
        
        # Exit if not running in continuous mode
        if not args.continuous:
            logger.log("All requirements processed. Exiting.")
            break