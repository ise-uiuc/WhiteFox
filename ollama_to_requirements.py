#!/usr/bin/env python3
"""
Convert Ollama generation outputs to requirements format expected by whitefox-torch-prompt-gen-req2test.sh

Usage:
python ollama_to_requirements.py --ollama-dir=MyRequirements --requirements-dir=MyRequirementsFormatted
"""

import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-dir", type=str, required=True,
                        help="Directory containing Ollama outputs")
    parser.add_argument("--requirements-dir", type=str, required=True, 
                        help="Output directory in requirements format")
    
    args = parser.parse_args()
    
    ollama_dir = Path(args.ollama_dir)
    req_dir = Path(args.requirements_dir)
    
    req_dir.mkdir(parents=True, exist_ok=True)
    
    for opt_dir in ollama_dir.iterdir():
        if not opt_dir.is_dir():
            continue
            
        opt_name = opt_dir.name
        
        opt_req_dir = req_dir / opt_name
        opt_req_dir.mkdir(exist_ok=True)
        
        py_file = opt_dir / f"{opt_name}_1.py"
        txt_file = opt_dir / f"{opt_name}_1.txt"
        
        if txt_file.exists():
            content = txt_file.read_text().strip()
            (opt_req_dir / f"{opt_name}_1.txt").write_text(content)
            print(f"Processed {opt_name}")
        elif py_file.exists():
            content = py_file.read_text().strip()
            (opt_req_dir / f"{opt_name}_1.txt").write_text(content)
            print(f"Processed {opt_name} (from .py file)")
        else:
            print(f"Warning: No output found for {opt_name}")

if __name__ == "__main__":
    main()
