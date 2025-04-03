"""
Name: auxiliary.py
Description: Contains auxiliary functions.
"""
import time
import os
import glob
import logging
import shutil
from pathlib import Path
from functools import wraps

#################################
### General utility functions ###
#################################

logger = logging.getLogger(__name__)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info("Starting %s...", func.__name__)
        result = func(*args, **kwargs)
        logger.info("Execution time for %s: %s seconds", func.__name__, time.time() - start_time)
        return result
    return wrapper

def log_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.debug(f"Args for {func.__name__}: {args}")
            logger.debug(f"Kwargs for {func.__name__}: {kwargs}")
            return None
    return wrapper


def skip_if_exists(output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        logger.debug("Skipping existing file: %s", output_path)
        return True
    return False

def delete_folder(folder_path: Path):
    logger.info("Deleting folder %s", folder_path)
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        logger.warning("Could not delete folder properly")
        logger.info("Deleted folder %s", folder_path)

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
