#!/usr/bin/env python3
"""
Convert Ollama outputs to requirements format expected by WhiteFox pipeline.

Usage:
python convert_ollama_to_requirements.py \
    --ollama-dir=MyRequirements \
    --requirements-dir=MyRequirementsFormatted
"""

import argparse
import json
import os
from pathlib import Path


def convert_ollama_to_requirements(ollama_dir, requirements_dir):
    """Convert Ollama outputs to the directory structure expected by the pipeline."""
    ollama_path = Path(ollama_dir)
    requirements_path = Path(requirements_dir)
    
    requirements_path.mkdir(parents=True, exist_ok=True)
    
    for opt_dir in ollama_path.iterdir():
        if not opt_dir.is_dir():
            continue
            
        opt_name = opt_dir.name
        print(f"Processing optimization: {opt_name}")
        
        req_opt_dir = requirements_path / opt_name
        req_opt_dir.mkdir(parents=True, exist_ok=True)
        
        text_files = list(opt_dir.glob("*.txt"))
        txt_files = [f for f in text_files if not f.name.startswith("prompt")]
        
        if not txt_files:
            print(f"Warning: No output text files found for {opt_name}")
            continue
            
        # Use the first generated output as the requirement
        # (Modify this logic if you want to combine multiple outputs)
        source_file = txt_files[0]
        
        target_file = req_opt_dir / f"{opt_name}_1.txt"
        
        content = source_file.read_text().strip()
        
        cleaned_content = clean_requirement_text(content)
        
        with open(target_file, 'w') as f:
            f.write(cleaned_content)
            
        print(f"  Created: {target_file}")


def clean_requirement_text(text):
    """Clean up the generated requirement text."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not cleaned_lines and not line:
            continue
            
        skip_prefixes = [
            "Assistant:",
            "AI:",
            "Response:",
            "Generated requirement:",
            "Requirement:",
        ]
        
        should_skip = False
        for prefix in skip_prefixes:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                if not line:
                    should_skip = True
                break
                
        if not should_skip and line:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    artifacts_to_remove = [
        "```",
        "###",
        "<|endoftext|>",
        "# Model ends",
    ]
    
    for artifact in artifacts_to_remove:
        if artifact in result:
            result = result.split(artifact)[0].strip()
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ollama outputs to requirements format")
    parser.add_argument("--ollama-dir", type=str, required=True,
                        help="Directory containing Ollama outputs")
    parser.add_argument("--requirements-dir", type=str, required=True,
                        help="Output directory for formatted requirements")
    
    args = parser.parse_args()
    
    print(f"Converting Ollama outputs from {args.ollama_dir} to {args.requirements_dir}")
    convert_ollama_to_requirements(args.ollama_dir, args.requirements_dir)
    print("Conversion completed!")
