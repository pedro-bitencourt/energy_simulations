"""
Name: auxiliary.py
Description: Contains auxiliary functions.
"""

import re
import os
import glob
import shutil
import copy
import pandas as pd
import subprocess
import logging

# Global variables
# Error codes
FILE_NOT_FOUND = -11
FILE_PATH_NOT_FOUND = -10
PATTERN_NOT_FOUND = -9
READING_ERROR = -12


def try_open_file(file_path, mode='r'):
    """
    Tries to open a file and return its content.
    """
    try:
        if 'b' in mode:  # Binary mode does not support encoding
            with open(file_path, mode) as file:
                content = file.read()
        else:
            with open(file_path, mode, encoding='iso-8859-1') as file:
                content = file.read()
        return content
    except FileNotFoundError:
        logging.error("File %s not found.", file_path)
        return False
    except Exception as unexpected_error:
        logging.error("File %s could not be read: %s",
                      file_path, unexpected_error)
        return False


def try_get_file(folder, filename, error_message=False, warn_if_multiple=False):
    """
    Tries to get file path.
    """
    if folder:
        file_path = glob.glob(os.path.join(folder, filename))
        if file_path:
            if len(file_path) > 1:
                logging.warning(
                    "More than one file located for %s in %s", filename, folder)
            return file_path[0]
        if error_message:
            logging.error("File %s not found in %s", filename, folder)
    return False


def is_simple_pattern(pattern):
    """
    Checks if pattern is simple or regular expression.
    """
    special_characters = set(".^$*+?{}[]|()\\")
    return not any(char in special_characters for char in pattern)


def simple_search(content, pattern):
    """
    Searches for a simple patterv in a file.
    """
    return 1 if pattern in content else 0


def regex_search(content, pattern, file_path=''):
    """
    Searches for a regular expression in a file.
    """
    match = re.search(pattern, content)
    if match:
        logging.info("Pattern %s found in %s", pattern, file_path)
        return match.group(1)
    logging.error("Pattern %s not found in %s", pattern, file_path)
    return PATTERN_NOT_FOUND


def find_pattern_in_file(file_path, pattern, error_message=False):
    """
    Finds a pattern in a file. If the pattern is a regex, return the matching
    gorup. If the pattern is simple, return 1 if found, 0 otherwise.
    """
    if file_path:
        content = try_open_file(file_path)
        if content:
            if is_simple_pattern(pattern):
                return simple_search(content, pattern)
            return regex_search(content, pattern, error_message, file_path)
        return FILE_NOT_FOUND
    return FILE_PATH_NOT_FOUND


def create_folder(base_path, experiment_name, remove_existing=False):
    """
    Creates a folder in the base path with the experiment name.
    """
    folder_path = os.path.join(base_path, experiment_name)
    # Check if folder already exists, remove if yes
    if os.path.exists(folder_path) and remove_existing:
        shutil.rmtree(folder_path)
        logging.info(f"Removed directory {folder_path}")
    try:
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"Successfully created the directory {folder_path}")
    except OSError as error:
        logging.error(
            f"Failed to create directory {folder_path}: {error.strerror}")
    return folder_path


def get_unique_filename(directory, base_name):
    """
    Determine the last file in a sequence and create a new file with an incremented numerical suffix.

    Args:
    - directory (str): The directory where the files are located.
    - base_name (str): The base name of the files.

    Returns:
    - str: The name of the newly created file.
    """
    pattern = re.compile(rf'{re.escape(base_name)}_(\d+)$')
    max_index = 0

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            max_index = max(max_index, int(match.group(1)))

    new_index = max_index + 1
    if new_index == 1:
        return f"{directory}/{base_name}"
    else:
        return f"{directory}/{base_name}_{new_index}"


def submit_slurm_job(script_path):
    """
    Submits a .sh file to SLURM and returns the job ID.

    Parameters:
    script_path (str): The path to the .sh file to be submitted.

    Returns:
    str: The job ID of the submitted job.
    """
    try:
        # Submit the job using sbatch
        result = subprocess.run(['sbatch', script_path],
                                capture_output=True, text=True, check=True)
        # Extract the job ID from the output
        output = result.stdout
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error("Error submitting job: %s", e.stderr)
        return None
