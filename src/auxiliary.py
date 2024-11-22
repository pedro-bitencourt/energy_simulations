"""
Name: auxiliary.py
Description: Contains auxiliary functions.
"""

import re
import os
import glob
import time
import subprocess
import logging
import numpy as np

#################################
### General utility functions ###
#################################


def convert_numpy_types(data: dict) -> dict:
    """
    Converts numpy data types in a dictionary to native Python types.

    Args:
        data (dict): The dictionary to convert.

    Returns:
        dict: The converted dictionary.
    """
    return {key: (int(value) if isinstance(value, np.integer)
                  else float(value) if isinstance(value, np.floating)
                  else value)
            for key, value in data.items()}


def make_name(float_list):
    """
    Takes list of floats, returns string with at most 2 decimals
    Example: [0.12345, 1.0, 2.67890] -> "0.123_1_2.679" 
    """
    formatted = []
    for num in float_list:
        # Format to 3 decimals and remove trailing zeros
        s = f"{num:.2f}".rstrip('0').rstrip('.')
        formatted.append(s)

    return "_".join(formatted)


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
    except Exception as unexpected_error:
        return False


def try_get_file(folder, filename):
    if folder:
        file_path = glob.glob(os.path.join(folder, filename))
        if file_path:
            if len(file_path) > 1:
                logging.warning(
                    "More than one file located for %s in %s", filename, folder)
            return file_path[0]
        logging.debug("File %s not found in %s", filename, folder)
        return False
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
        return match.group(1)
    logging.error("Pattern %s not found in %s", pattern, file_path)
    return None


def find_pattern_in_file(file_path, pattern):
    """
    Finds a pattern in a file. If the pattern is a regex, return the matching
    gorup. If the pattern is simple, return 1 if found, 0 otherwise.
    """
    if file_path:
        content = try_open_file(file_path)
        if content:
            if is_simple_pattern(pattern):
                return simple_search(content, pattern)
            return regex_search(content, pattern, file_path)
        return None
    return None


#################################
### SLURM utility functions ###
#################################
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


def check_job_status(job_id: int):
    '''
    Check the status of a job in SLURM.
    '''
    result = subprocess.run(
        ['squeue', '-j', str(job_id), '-h'], capture_output=True, text=True)
    return len(result.stdout.strip()) > 0


def wait_for_jobs(job_ids: list):
    '''
    Wait for a list of jobs to finish.
    '''
    job_ids = [job_id for job_id in job_ids if job_id is not True]
    while True:
        if all(not check_job_status(job_id) for job_id in job_ids):
            break
        time.sleep(60)
