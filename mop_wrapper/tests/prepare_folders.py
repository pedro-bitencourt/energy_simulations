import os
from pathlib import Path
import sys

def truncate_xlt_file(file_path, num_lines=1000):
    # Read first num_lines lines
    with open(file_path, 'r') as f:
        lines = [next(f) for _ in range(num_lines)]
    
    # Write back to same file
    with open(file_path, 'w') as f:
        f.writelines(lines)

def process_directory(directory):
    # Convert to Path object
    dir_path = Path(directory)
    
    # Walk through all files and subdirectories
    for path in dir_path.rglob('*.xlt'):
        print(f"Processing {path}")
        try:
            truncate_xlt_file(path)
            print(f"Successfully truncated {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)
        
    process_directory(directory)
