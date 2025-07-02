#!/bin/bash
# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Test counter variables
passed=0
failed=0
skipped=0
total=0

# Clear previous results
> test_results.log
echo "Starting Prolog tests..."

# Check if ws.pl exists
if [ ! -f "ws.pl" ]; then
    echo -e "${RED}ERROR: ws.pl not found in current directory${NC}"
    exit 1
fi

# Check if swipl-win.exe is available (try multiple possible names)
SWIPL_CMD=""
if command -v swipl-win.exe &> /dev/null; then
    SWIPL_CMD="swipl-win.exe"
elif command -v swipl.exe &> /dev/null; then
    SWIPL_CMD="swipl.exe"
elif command -v swipl &> /dev/null; then
    SWIPL_CMD="swipl"
else
    echo -e "${RED}ERROR: SWI-Prolog not found in PATH${NC}"
    echo "Please ensure SWI-Prolog is installed and in your PATH"
    echo "Tried: swipl-win.exe, swipl.exe, swipl"
    exit 1
fi

echo "Using SWI-Prolog command: $SWIPL_CMD"

# Loop through all input files from 001 to 260
for i in $(seq -f "%03g" 1 320); do
    input_file="input-${i}.txt"
    output_file="output-${i}.txt"
    expected_file="expected-${i}.txt"
    
    # Determine input path
    if [ -f "${input_file}" ]; then
        input_path="${input_file}"
    elif [ -f "input/${input_file}" ]; then
        input_path="input/${input_file}"
    else
        continue  # Skip if input file not found in either location
    fi
    
    # Run the Prolog script
    echo "Running test case ${i}..." >&2
    
    # Use timeout if available, otherwise run without timeout
    if command -v timeout &> /dev/null; then
        timeout 30s $SWIPL_CMD -l ws.pl -g main "${input_path}" 2>/dev/null
    else
        $SWIPL_CMD -l ws.pl -g main "${input_path}" 2>/dev/null
    fi
    exit_code=$?
    
    # Check if Prolog execution timed out or failed
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}Test case ${i}: TIMEOUT (Prolog execution exceeded 30 seconds)${NC}"
        echo "Test case ${i}: TIMEOUT" >> test_results.log
        total=$((total+1))
        failed=$((failed+1))
        # Clean up output file if it exists
        [ -f "${output_file}" ] && rm -f "${output_file}"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo -e "${RED}Test case ${i}: ERROR (Prolog execution failed with exit code ${exit_code})${NC}"
        echo "Test case ${i}: ERROR (Prolog execution failed)" >> test_results.log
        total=$((total+1))
        failed=$((failed+1))
        # Clean up output file if it exists
        [ -f "${output_file}" ] && rm -f "${output_file}"
        continue
    fi
    
    # Post-process the output file (always in current directory)
    if [ -f "${output_file}" ]; then
        # Remove carriage returns and trailing whitespace (Git Bash compatible)
        if command -v sed &> /dev/null; then
            # Use sed if available
            sed -i.bak 's/\r$//' "${output_file}" && rm -f "${output_file}.bak"
            sed -i.bak 's/[ \t]*$//' "${output_file}" && rm -f "${output_file}.bak"
        else
            # Fallback method using tr and other basic tools
            tr -d '\r' < "${output_file}" > "${output_file}.tmp" && mv "${output_file}.tmp" "${output_file}"
        fi
        
        # Ensure file ends with newline
        if [ -s "${output_file}" ]; then
            if [ "$(tail -c1 "${output_file}" | wc -l)" -eq 0 ]; then
                echo >> "${output_file}"
            fi
        fi
    else
        echo -e "${RED}Test case ${i}: ERROR (Output file not generated)${NC}"
        echo "Test case ${i}: ERROR (Output file not generated)" >> test_results.log
        total=$((total+1))
        failed=$((failed+1))
        continue
    fi
    
    total=$((total+1))
    
    # Check if expected file exists
    if [ ! -f "expected/${expected_file}" ]; then
        echo -e "${YELLOW}Test case ${i}: SKIPPED (Expected file missing)${NC}"
        echo "Test case ${i}: SKIPPED (Expected file missing)" >> test_results.log
        skipped=$((skipped+1))
        # Clean up output file since we can't compare it
        rm -f "${output_file}"
        continue
    fi
    
    # Compare output vs expected
    if diff -q --ignore-trailing-space "${output_file}" "expected/${expected_file}" >/dev/null 2>&1; then
        echo -e "${GREEN}Test case ${i}: PASSED${NC}"
        echo "Test case ${i}: PASSED" >> test_results.log
        passed=$((passed+1))
    else
        echo -e "${RED}Test case ${i}: FAILED${NC}"
        echo "Test case ${i}: FAILED" >> test_results.log
        echo "Differences:" >> test_results.log
        echo "----------------------" >> test_results.log
        echo "Expected:" >> test_results.log
        cat "expected/${expected_file}" >> test_results.log
        echo "----------------------" >> test_results.log
        echo "Generated:" >> test_results.log
        cat "${output_file}" >> test_results.log
        echo "----------------------" >> test_results.log
        failed=$((failed+1))
        
        # Display differences for debugging
        echo "Differences for test case ${i}:"
        echo "----------------------"
        echo "Expected:"
        cat "expected/${expected_file}"
        echo "----------------------"
        echo "Generated:"
        cat "${output_file}"
        echo "----------------------"
    fi
    
    # Clean up output file after comparison
    rm -f "${output_file}"
done

# Print summary
echo -e "\nProlog Test Summary:"
echo -e "${GREEN}Passed: ${passed}${NC}"
echo -e "${RED}Failed: ${failed}${NC}"
echo -e "${YELLOW}Skipped: ${skipped}${NC}"
echo -e "Total: ${total}"

# Save summary to log
echo -e "\nProlog Test Summary:" >> test_results.log
echo "Passed: ${passed}" >> test_results.log
echo "Failed: ${failed}" >> test_results.log
echo "Skipped: ${skipped}" >> test_results.log
echo "Total: ${total}" >> test_results.log

echo "Test results saved to test_results.log"

# Exit with appropriate code
if [ $failed -gt 0 ]; then
    exit 1
else
    exit 0
fi