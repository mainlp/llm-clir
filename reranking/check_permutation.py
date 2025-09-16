import re
import argparse

def check_permutation_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    errors = []
    total_permutations = 0
    correct_permutations = 0
    permutation_pattern = re.compile(r'(\[\d+\] > )+\[\d+\]$') 
    number_pattern = re.compile(r'\d+')

    for idx, line in enumerate(lines):
        line = line.strip()

        if not line.startswith("Permutation after running LLM:"):
            continue  
        
        total_permutations += 1
        
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        
        if not permutation_pattern.fullmatch(next_line):
            errors.append(f"Line {idx+2}: wrong format -> {next_line}")
            continue  

        # extract all numbers
        numbers = list(map(int, number_pattern.findall(next_line)))

        # check numbers
        if set(numbers) != set(range(1, 17)):
            correct_permutations += 1 
        else:
            errors.append(f"Line {idx+2}: index missing or out of range -> {numbers}")
    
    accuracy = (correct_permutations / total_permutations) * 100 if total_permutations > 0 else 0
    
    print(f"Total permutation: {total_permutations}")
    print(f"Correct permutation: {correct_permutations}")
    print(f"Success Rate: {accuracy:.2f}%")
    
    if errors:
        print("Errors Detected:")
        for error in errors:
            print(error)
    else:
        print("All Permutations are correct!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation Output Check")
    parser.add_argument("file_path", type=str, help="path to permutation log file")
    args = parser.parse_args()
    
    check_permutation_file(args.file_path)