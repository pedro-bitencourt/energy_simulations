
import subprocess
import time
from pathlib import Path


SLURM_DEFAULTS = {
    'run': {
        'time': 0.8,
        'memory': 5,
        'email': None,
        'mail_type': 'NONE'
    },
    'solver': {
        'time': 12,
        'memory': 5,
        'email': None,
        'mail_type': 'END,FAIL'
    },
    'processing': {
        'time': 5,
        'memory': 5,
        'email': None,
        'mail_type': 'END,FAIL'
    }
}

MEMORY_DEFAULT = 5  # GB


def format_requested_time(requested_time: float) -> str:
    hours: int = int(requested_time)
    minutes: int = int((requested_time * 60) % 60)
    seconds: int = int((requested_time * 3600) % 60)
    requested_time_run: str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return requested_time_run


def slurm_header(slurm_configs: dict, job_name: str, job_type: str,
                 slurm_path: Path, email=None):
    if not slurm_configs:
        slurm_config = SLURM_DEFAULTS[job_type]
        slurm_configs = {job_type: slurm_config}
    slurm_config: dict = construct_slurm_config(slurm_configs, job_type)

    requested_time: float = slurm_config['time']
    requested_time_str: str = format_requested_time(requested_time)
    email_line: str = f"#SBATCH --mail-user={email}" if email else ""
    mail_type_line: str = f"#SBATCH --mail-type={slurm_config['mail_type']}" if email else ""
    memory: int = slurm_config.get('memory', MEMORY_DEFAULT)

    header = f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time_str}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem={memory}G
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_path.parent}/slurm.out
#SBATCH --error={slurm_path.parent}/slurm.err
#SBATCH --exclude=qhimem[0207-0208],qnode0255,qnode0257,qnode0260
{email_line}
{mail_type_line}
'''
    return header


def construct_slurm_config(slurm_configs: dict, job_type: str) -> dict:
    slurm_config: dict = slurm_configs.get(job_type, {})
    default_config: dict = SLURM_DEFAULTS[job_type]
    slurm_config: dict = slurm_config | default_config
    return slurm_config


def get_job_id_by_name(job_name: str):
    """Check if a job with given name exists and return its ID."""
    try:
        result = subprocess.run(['squeue', '-h', '-n', job_name, '-o', '%i'],
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def submit_new_job(script_path: str):
    """Submit a new SLURM job with given name and return its ID."""
    result = subprocess.run(['sbatch', script_path],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip().split()[-1]


def submit_slurm_job(script_path: str, job_name: str):
    """
    Submit a SLURM job or return ID of existing job with same name.

    Parameters:
        script_path (str): Path to the .sh script to submit
        job_name (str): Name for the job

    Returns:
        str: Job ID if successful
        None: If submission failed and no existing job found
    """
    # Check for existing job
    if existing_id := get_job_id_by_name(job_name):
        return existing_id

    # Submit new job if none exists
    return submit_new_job(script_path)


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
