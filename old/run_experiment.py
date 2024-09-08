"""
File name: run_experiment.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: This script submits multiple jobs using the MOP software. 
"""
import subprocess
import os
import shutil
import glob
import re
import numpy as np
import extract_results

# Define global variables
# Set base path
BASE_PATH = r'/projects/p32342'
# Set location of slurm template
RUN_SLURM_TEMPLATE = '/projects/p32342/code/aux/run_template.sh'
# Set location of master job slurm template
MASTER_SLURM_TEMPLATE = '/projects/p32342/code/aux/master_template.sh'
# Set location of temporary master slurm
slurm_script_path = f'{BASE_PATH}/code/aux/master.sh'
# Rerun script path
rerun_script_path = f'{BASE_PATH}/code/aux/rerun.sh'
# Process script path
process_script_path = f'{BASE_PATH}/code/aux/process.sh'

# Define experiment class
class Experiment:
    """
    Represents an experiment. Creates multiple runs from a set of values and a
    .xml basefile.
    """
    def __init__(self, name, variables, set_of_values, pattern_variables,
            xml_basefile):
        self.name = name
        self.variables = variables
        self.set_of_values = set_of_values
        self.pattern_variables = pattern_variables
        self.xml_basefile = xml_basefile
        self.name_subfolder, self.initial_year, self.final_year = self.get_parameters()
        self.runs_array = []
        ###### Create relevant folders
        # Create input folder
        self.input_path = create_folder(f"{BASE_PATH}/temp", self.name, remove_existing = False)
        # Create output folder
        self.output_path = create_folder(fr"{BASE_PATH}/output", self.name, remove_existing = False)
        # Create results folder
        self.result_path = create_folder(fr"{BASE_PATH}/results", self.name, remove_existing = False)
        for value in set_of_values:
            self.runs_array.append(Run(self, value))

    def get_parameters(self):
        """
        Extract name of the run, the initial year and the final year
        from the .xml file
        """
        with open(self.xml_basefile, 'r', encoding='utf-8') as file:
            content = file.read()
            name =content.split("<nombre>")[1].split("</nombre>")[0].strip()
            initial_year = int(content.split("<inicioCorrida>")[1].split(" ")[2])
            final_year = int(content.split("<finCorrida>")[1].split(" ")[2])
        return name, initial_year, final_year

# Define run class
class Run:
    """
    Representes a run. Creates a .xml file from the .xml template.
    """
    def __init__(self, experiment, values, folders = None):
        self.values = values
        self.run_name = self.get_run_name()
        self.xml_file = self.create_xml(experiment)
        self.output_path = f"{experiment.output_path}/{self.run_name}"
        self.output_path_complete = f"{experiment.output_path}/{self.run_name}/{experiment.name_subfolder}" 
        self.slurm_output = fr'{experiment.output_path}/{self.run_name}.out'
        self.folders = folders

    # get_run_name: this function creates a unique name for each run
    def get_run_name(self):
        """
        Generates the run name.
        """
        string = ""
        first_var = True
        for v in self.values:
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            if first_var:
                string = f"{string}{v}"
                first_var = False
            else:
                string = f"{string}_{v}"
        return string
    
	# create_xml: this function edits a base .xml setting located in base_file. 
    def create_xml(self, experiment):
        #run_name = get_run_name(value)
        file_name = f"{experiment.input_path}/{self.run_name}.xml"
        shutil.copy(experiment.xml_basefile, file_name)
        # Create .xml configuration file
        with open(file_name, 'r') as file:
            # Open base file
            content = file.read()
            # Change output path
            content = re.sub(fr'<rutaSalidas>.*?</rutaSalidas>',
                    fr'<rutaSalidas>Z:\\projects\\p32342\\output\\{experiment.name}\\{self.run_name}</rutaSalidas>',
                    content)
            # Change variables
            for idx in range(len(self.values)):
                content = re.sub(experiment.pattern_variables[idx],f"{self.values[idx]}", content)
        with open(file_name, 'w') as file:
            # Write new file
            file.write(content)
        return file_name

def run_experiment(experiment):
    job_ids = []
    for run in experiment.runs_array:    
        job_id = submit_run(run)
        job_ids.append(job_id)
    print("All jobs submitted. Submitting master job...")
    submit_master_job(job_ids, experiment)
    

##### RERUN FUNCTIONS
def rerun_experiment(experiment, run = True):
    for run in experiment.runs_array:
        run.folders = get_folders(run)
    general_results = extract_results.get_general_results(experiment)
    #failed_runs = [result[variables] for result in general_results if result['successful'] == 0]
    failed_runs = general_results.loc[general_results['successful'] == 0, experiment.variables]
    failed_runs = failed_runs.values
    failed_runs = np.array([row for row in failed_runs])
    print("Failed runs:\n")
    print(failed_runs)
    print("\nNumber of failed runs:\n")
    print(failed_runs.shape[0])
    rr_experiment =  Experiment(experiment.name, experiment.variables, failed_runs, 
                                experiment.pattern_variables, experiment.xml_basefile)
    if run:
        run_experiment(rr_experiment)
    else:
        return rr_experiment

def get_folders(run):
    temp = ["*-OPT","*-SIM"]
    folders = [None]*2
    for i in range(2):
        folders[i] = glob.glob(os.path.join(run.output_path_complete, temp[i]))
        if not folders[i]:
            print(f"Folder {run.output_path_complete}/{temp[i]} not found")
            return folders
    return {'OPT': folders[0][0], 'SIM': folders[1][0]}

# Submit a master job
def submit_master_job(job_ids, experiment):
    with open(f'/projects/p32342/temp/{experiment.name}.txt', 'r') as file: 
        tries_left = int(file.read()) 
    job_ids_string = ":".join(job_ids)
    print(job_ids_string)
    # Submit the job
    if tries_left > 0:
        script_path = '/projects/p32342/code/aux/rerun.sh'
        with open(f'/projects/p32342/temp/{experiment.name}.txt', 'w') as file:
            file.write(str(tries_left-1))
    else:
        script_path = '/projects/p32342/code/aux/process.sh'
    
    # Construct the command
    command = [
            "sbatch",
                "--dependency=afterany:{}".format(job_ids_string),
                    script_path
                    ]

                # Print the command to verify it's correct (optional)
    print("Running command:", " ".join(command))
    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the output and error
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)
    
def submit_run(run):
    print(f"Preparing to submit job {run.xml_file}")
    # Create a unique slurm script for each job
    filename = re.sub(r'.xml$', '', run.xml_file)
    slurm_script_path = f'{filename}.sh'
    with open(RUN_SLURM_TEMPLATE,'r') as template:
        content = template.read()
        content = re.sub('<OUTPUTFILENAME>', run.slurm_output,content)
        content = re.sub('<JOBNAME>',run.xml_file,content)
        content = re.sub(r'<OUTPUTFOLDERPATH>', run.output_path, content)
        xml_file = run.xml_file.replace("/","\\")
        # Escape special characters except the .xml extension
        xml_file = re.escape(xml_file[:-4]) + '.xml'
        content = re.sub(r'<XMLFILENAME>',xml_file,content)
    with open(slurm_script_path, 'w') as slurm_script:
        slurm_script.write(content)
    # Submit the job
    temp_output_file =  f'{BASE_PATH}/temp/temp.txt' 
    command = f"sbatch {slurm_script_path} > {temp_output_file} 2>&1"
    return_code = os.system(command)
    if return_code != 0:
        print(f"Error: sbatch failed with return code {return_code}")
        raise RuntimeError("sbatch command failed")

    # Read the output from the temporary file
    with open(temp_output_file, 'r') as file:
        output = file.read().strip()

    # Clean up the temporary file
    os.remove(temp_output_file)

    if output:
        # Extract the job ID from the output
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            print(f"Successfully submitted run {run.run_name} with jobid {job_id}")
            return job_id
        else:
            print(f"Unexpected output: {output}")
            raise RuntimeError("Unexpected output from sbatch command")
    else:
        print("Error: No output from sbatch command")
        raise RuntimeError("No output from sbatch command") 

# Auxiliary function to create folders
def create_folder(base_path, experiment_name, remove_existing):
    folder_path = os.path.join(base_path, experiment_name)
    # Check if folder already exists, remove if yes
    if os.path.exists(folder_path) and remove_existing:
        shutil.rmtree(folder_path)
        print(f"Removed directory {folder_path}")
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Successfully created the directory {folder_path}")
    except OSError as error:
        print(f"Failed to create directory {folder_path}: {error.strerror}")
    return folder_path
