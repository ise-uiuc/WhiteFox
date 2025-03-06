#!/usr/bin/env python3
"""
Script to run starcoder_ollama.py with time and file count limits.
Will run for either 10 hours or until 200 .c files are generated, whichever comes first.
"""

import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
COMMAND = "python3 starcoder_ollama.py --requirements-dir=llvm-exec/source-code-data/llvm/llvm-gen-prompt/requirements --output-dir=test --num=1"
MAX_RUNTIME_HOURS = 10
MAX_FILES = 200
OUTPUT_DIR = "test"  # Must match the --output-dir parameter in COMMAND

def count_c_files(directory):
    """Count the number of .c files in the specified directory"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob("*.c")))

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate end time
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=MAX_RUNTIME_HOURS)
    
    print(f"Starting script at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will run until {end_time.strftime('%Y-%m-%d %H:%M:%S')} or until {MAX_FILES} .c files are generated")
    print(f"Command: {COMMAND}")
    
    # Main loop
    iteration = 1
    while True:
        current_time = datetime.now()
        
        # Check time limit
        if current_time >= end_time:
            print(f"Reached time limit of {MAX_RUNTIME_HOURS} hours. Stopping.")
            break
        
        # Check file count
        file_count = count_c_files(OUTPUT_DIR)
        print(f"Iteration {iteration}: Current file count: {file_count}/{MAX_FILES}")
        
        if file_count >= MAX_FILES:
            print(f"Reached file limit of {MAX_FILES} .c files. Stopping.")
            break
        
        # Calculate and display remaining time
        time_elapsed = current_time - start_time
        time_remaining = end_time - current_time
        print(f"Time elapsed: {time_elapsed}, Time remaining: {time_remaining}")
        
        # Run the command
        print(f"Running iteration {iteration}...")
        try:
            process = subprocess.run(COMMAND, shell=True, check=True, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     universal_newlines=True)
            
            # Print output (optional - comment out if too verbose)
            print("Command output:")
            print(process.stdout)
            
            if process.stderr:
                print("Command errors:")
                print(process.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Exit code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            
            # Optional: add a delay before retrying after error
            print("Waiting 30 seconds before trying again...")
            time.sleep(30)
        
        # Increment iteration counter
        iteration += 1
        
        # Optional: small delay between iterations to prevent system overload
        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
        sys.exit(0)