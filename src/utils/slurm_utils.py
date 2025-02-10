
import subprocess
import time
import logging
from pathlib import Path

MEMORY_DEFAULT = 5  # GB


def slurm_header(slurm_config: dict, job_name: str, slurm_path: Path) -> str:
    """
    Creates a bash file to be submitted to the cluster.
    """
    slurm_path = str(slurm_path)
    requested_time: float = slurm_config['time']
    hours: int = int(requested_time)
    minutes: int = int((requested_time * 60) % 60)
    seconds: int = int((requested_time * 3600) % 60)
    requested_time_run: str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    email = slurm_config.get('email', None)
    if email:
        email_line = f"#SBATCH --mail-user={email}"
    else:
        email_line = ""
    mail_type = slurm_config.get('mail-type', 'NONE')
    if mail_type and email:
        mail_type_line = f"#SBATCH --mail-type={mail_type}"
    else:
        mail_type_line = ""
    memory = slurm_config.get('memory', MEMORY_DEFAULT)

    header = f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time_run}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem={memory}G
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_path}/slurm.out
#SBATCH --error={slurm_path}/slurm.err
#SBATCH --exclude=qhimem[0207-0208],qnode0255,qnode0257,qnode0260
{email_line}
{mail_type_line}
'''
    return header


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
    try:
        result = subprocess.run(['sbatch', script_path],
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e.stderr}")
        return None


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
