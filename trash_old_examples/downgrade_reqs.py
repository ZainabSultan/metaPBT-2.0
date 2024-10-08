import os
import sys

def modify_requirements_file(file_path, output_path=None):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        modified_line = line.replace('==', '<=')
        modified_lines.append(modified_line)

    if output_path:
        with open(output_path, 'w') as file:
            file.writelines(modified_lines)
        print(f"Modified requirements saved to {output_path}")
    else:
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
        print(f"Modified requirements saved to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_requirements.py <path_to_requirements.txt> [output_file]")
        sys.exit(1)

    requirements_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    modify_requirements_file(requirements_path, output_path)
