#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import difflib
from pathlib import Path
import time

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
NC = '\033[0m'  # No Color

# Test counter variables
passed = 0
failed = 0
skipped = 0
total = 0

# Clear previous results
with open('test_results.log', 'w') as f:
    pass

print("Starting Prolog tests...")

# Check if ws.pl exists
if not os.path.exists('ws.pl'):
    print(f"{RED}ERROR: ws.pl not found in current directory{NC}")
    sys.exit(1)

# Find SWI-Prolog command
SWIPL_CMD = None
for cmd in ['swipl-win.exe', 'swipl.exe', 'swipl']:
    if shutil.which(cmd):
        SWIPL_CMD = cmd
        break

if not SWIPL_CMD:
    print(f"{RED}ERROR: SWI-Prolog not found in PATH{NC}")
    print("Please ensure SWI-Prolog is installed and in your PATH")
    print("Tried: swipl-win.exe, swipl.exe, swipl")
    sys.exit(1)

print(f"Using SWI-Prolog command: {SWIPL_CMD}")

# Function to append to log file
def log_message(message):
    with open('test_results.log', 'a') as f:
        f.write(message + '\n')

# Function to normalize line endings
def normalize_file(filepath):
    """Remove carriage returns and trailing whitespace, ensure file ends with newline"""
    if not os.path.exists(filepath):
        return
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Remove carriage returns and trailing whitespace
    lines = [line.rstrip() + '\n' for line in lines]
    
    # Remove empty trailing newlines but keep one
    while len(lines) > 1 and lines[-1].strip() == '':
        lines.pop()
    
    # Ensure file ends with exactly one newline
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Function to compare files
def files_are_equal(file1, file2):
    """Compare two files ignoring trailing whitespace"""
    try:
        with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
            lines1 = [line.rstrip() for line in f1.readlines()]
        with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
            lines2 = [line.rstrip() for line in f2.readlines()]
        return lines1 == lines2
    except:
        return False

# Loop through all input files from 001 to 320
for i in range(1, 321):
    input_file = f"input-{i:03d}.txt"
    output_file = f"output-{i:03d}.txt"
    expected_file = f"expected-{i:03d}.txt"
    
    # Determine input path
    input_path = None
    if os.path.exists(input_file):
        input_path = input_file
    elif os.path.exists(f"input/{input_file}"):
        input_path = f"input/{input_file}"
    else:
        continue  # Skip if input file not found
    
    # Run the Prolog script
    print(f"Running test case {i:03d}...", file=sys.stderr)
    
    try:
        # Run with timeout of 30 seconds
        result = subprocess.run(
            [SWIPL_CMD, '-l', 'ws.pl', '-g', 'main', input_path],
            capture_output=True,
            timeout=30
        )
        exit_code = result.returncode
        timeout_occurred = False
    except subprocess.TimeoutExpired:
        exit_code = 124
        timeout_occurred = True
    except Exception as e:
        exit_code = -1
        timeout_occurred = False
    
    # Check if Prolog execution timed out or failed
    if timeout_occurred:
        print(f"{RED}Test case {i:03d}: TIMEOUT (Prolog execution exceeded 30 seconds){NC}")
        log_message(f"Test case {i:03d}: TIMEOUT")
        total += 1
        failed += 1
        # Clean up output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        continue
    elif exit_code != 0:
        print(f"{RED}Test case {i:03d}: ERROR (Prolog execution failed with exit code {exit_code}){NC}")
        log_message(f"Test case {i:03d}: ERROR (Prolog execution failed)")
        total += 1
        failed += 1
        # Clean up output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        continue
    
    # Post-process the output file
    if os.path.exists(output_file):
        normalize_file(output_file)
    else:
        print(f"{RED}Test case {i:03d}: ERROR (Output file not generated){NC}")
        log_message(f"Test case {i:03d}: ERROR (Output file not generated)")
        total += 1
        failed += 1
        continue
    
    total += 1
    
    # Check if expected file exists
    if not os.path.exists(f"expected/{expected_file}"):
        print(f"{YELLOW}Test case {i:03d}: SKIPPED (Expected file missing){NC}")
        log_message(f"Test case {i:03d}: SKIPPED (Expected file missing)")
        skipped += 1
        # Clean up output file since we can't compare it
        os.remove(output_file)
        continue
    
    # Compare output vs expected
    if files_are_equal(output_file, f"expected/{expected_file}"):
        print(f"{GREEN}Test case {i:03d}: PASSED{NC}")
        log_message(f"Test case {i:03d}: PASSED")
        passed += 1
    else:
        print(f"{RED}Test case {i:03d}: FAILED{NC}")
        log_message(f"Test case {i:03d}: FAILED")
        log_message("Differences:")
        log_message("----------------------")
        log_message("Expected:")
        with open(f"expected/{expected_file}", 'r', encoding='utf-8', errors='ignore') as f:
            log_message(f.read())
        log_message("----------------------")
        log_message("Generated:")
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_message(f.read())
        log_message("----------------------")
        failed += 1
        
        # Display differences for debugging
        print(f"Differences for test case {i:03d}:")
        print("----------------------")
        print("Expected:")
        with open(f"expected/{expected_file}", 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read())
        print("----------------------")
        print("Generated:")
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read())
        print("----------------------")
    
    # Clean up output file after comparison
    os.remove(output_file)

# Print summary
print(f"\nProlog Test Summary:")
print(f"{GREEN}Passed: {passed}{NC}")
print(f"{RED}Failed: {failed}{NC}")
print(f"{YELLOW}Skipped: {skipped}{NC}")
print(f"Total: {total}")

# Save summary to log
log_message("\nProlog Test Summary:")
log_message(f"Passed: {passed}")
log_message(f"Failed: {failed}")
log_message(f"Skipped: {skipped}")
log_message(f"Total: {total}")

print("Test results saved to test_results.log")

# Exit with appropriate code
sys.exit(1 if failed > 0 else 0)