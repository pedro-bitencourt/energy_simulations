"""
File name: run_module.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: this file implements the Run class and related methods.
"""

import os
import re
from typing import Optional
from pathlib import Path
import auxiliary
import logging


class Run:
    """
    Representes a run and implements some general purpose methods for it.
    """

    def __init__(self,
                 input_folder: Path,
                 general_parameters: dict,
                 variables: dict[str, dict],
                 output_folder: Optional[Path] = None,
                 experiment_folder: Optional[Path] = None):
        self.variables: dict[str, dict] = variables
        self.values: list = [entry['value'] for entry in variables.values()]
        self.general_parameters: dict = general_parameters

        # initialize the run name and output folder
        if output_folder:
            self.run_name: str = Path(output_folder).name
        elif experiment_folder:
            run_name_function_params: dict = general_parameters['name_function']

            def run_name_from_params(params: dict, variables: dict):
                name: str = ''
                for key, entry in params.items():
                    name += f"{int(entry['multiplier']*variables[key]['value'])}_"
                return name.rstrip('_')
            self.run_name: str = run_name_from_params(
                run_name_function_params, self.variables)
            output_folder = experiment_folder / self.run_name
        else:
            print("CRITICAL: no output folder or experiment folder provided.")

        # initialize relevant paths
        self.paths: dict = self.initialize_paths(
            input_folder, output_folder, general_parameters)

    def __repr__(self):
        return f"Run(run_name={self.run_name}, paths = {self.paths}, successful = {self.successful()}"

    def initialize_paths(self, input_folder, output_folder, general_parameters):
        """
        Initialize a dictionary with relevant paths for the run.
        """
        def format_windows_path(path_str):
            # replace forward slashes with backslashes
            windows_path = path_str.replace('/', '\\')

            # add Z: at the start of the path
            windows_path = 'Z:' + windows_path

            # if the path doesn't end with a backslash, add one
            if not windows_path.endswith('\\'):
                windows_path += '\\'

            return windows_path

        run_paths = {}
        run_paths['output'] = str(output_folder)
        # convert the output path to a windows path, for use in the .xml file
        run_paths['output_windows'] = format_windows_path(run_paths['output'])

        subfolder = general_parameters['name_subfolder']
        run_paths['output_complete'] = f"{output_folder}/{subfolder}"
        run_paths['input'] = Path(f"{input_folder}")
        run_paths['results_json'] = Path(
            f"{output_folder}/{self.run_name}_results.json")
        run_paths['price_distribution'] = Path(
            f"{output_folder}/price_distribution.csv")
        return run_paths

    def get_opt_and_sim_folders(self):
        '''
        Get the opt and sim folders from the output_complete path.
        '''
        temp_paths = {}
        temp_dict = {'opt': "*-OPT", 'sim': "*-SIM"}
        test_files = {'opt': r'tiempos*', 'sim': r'resumIng*'}
        output_complete_path = Path(self.paths['output_complete'])
        for key, value in temp_dict.items():
            temp_paths[key] = None
            temp_folder_list = list(output_complete_path.glob(value))
            if temp_folder_list:
                for folder in temp_folder_list:
                    correct_folder = auxiliary.try_get_file(
                        folder, test_files[key])
                    if correct_folder:
                        temp_paths[key] = folder
                        continue
        return temp_paths

    def successful(self):
        """
        Check if the run was successful by searching for a res* file
        in the sim folder.
        """
        self.paths.update(self.get_opt_and_sim_folders())
        sim_folder = self.paths.get('sim', False)
        if sim_folder:
            resumen_file = auxiliary.try_get_file(sim_folder, r'resumen*')
            if resumen_file:
                return True
        else:
            logging.error('No sim folder found for run %s in folder %s',
                          self.run_name,
                          self.paths['output_complete'])
        return False

    def submit_run(self, run_time=None):
        """
        Submits a run to the cluster.
        """
        # break if already successful
        if self.successful():
            print(f"Run {self.run_name} already successful, skipping.")
            return True

        print(f"""Preparing to submit run {self.run_name},
                    with successful status {self.successful()}""")

        # create the RunSubmitter object
        submitter = RunSubmitter(self, run_time)

        # submit the run
        job_id = submitter.submit_run()

        return job_id


###############################################################################
# SUBMITTING RUN
###############################################################################

class RunSubmitter:
    def __init__(self, run: Run, run_time=None):
        self.run: Run = run
        self.run_time: str = run_time
        self.paths: dict = self.update_paths(run.paths)
        # create the .xml and the .sh files
        self.create_xml(run.general_parameters['xml_basefile'])
        self.create_bash_script()

    def update_paths(self, paths: dict):
        input_folder = paths['input']
        output_folder = paths['output']

        paths['xml'] = f"{input_folder}/{self.run.run_name}.xml"
        paths['slurm_script'] = f"{input_folder}/{self.run.run_name}.sh"
        paths['slurm_output'] = f"{output_folder}/{self.run.run_name}.out"
        return paths

    def create_xml(self, xml_basefile: Path):
        """
        Creates a .xml file from the .xml template.
        """
        # open the experiment's .xml basefile
        with open(xml_basefile, 'r') as file:
            content = file.read()

            def escape_for_regex(path):
                return path.replace('\\', '\\\\')

            # change output path
            content = re.sub(
                r'<rutaSalidas>.*?</rutaSalidas>',
                f'<rutaSalidas>{escape_for_regex(self.paths["output_windows"])}</rutaSalidas>',
                content
            )

            # substitute variables values
            content = self.substitute_variables(content)

        # write the new settings to the .xml file path
        with open(self.paths['xml'], 'w') as file:
            file.write(content)

    def substitute_variables(self, content):
        for variable in self.run.variables.values():
            pattern = variable['pattern']
            value = variable['value']

            def replace_func(match):
                if '*' in match.group():
                    multiplier = float(match.group().split('*')[1].rstrip('<'))
                    new_value = int(value * multiplier)
                    return f">{str(new_value)}<"
                return f'>{str(value)}<'
            content = re.sub(
                f'>{pattern}(?:\*[\d.]+)?<', replace_func, content)
        return content

    def create_bash_script(self, run_time=None):
        script_path = self.paths['slurm_script']
        xml_path = os.path.normpath(self.paths['xml'])
        xml_path = xml_path.replace(os.path.sep, '\\')

        if run_time is None:
            run_time = '2:30:00'

        with open(script_path, 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={run_time}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=5G 
#SBATCH --job-name={self.run.run_name}
#SBATCH --output={self.paths['slurm_output']}
#SBATCH --error={self.paths['slurm_output']}.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
sleep $((RANDOM%60 + 10)) 
wine "Z:\projects\p32342\software\Java\jdk-11.0.22+7\\bin\java.exe" -Xmx5G -jar MOP_Mingo.JAR "Z:{xml_path}"
''')

    def submit_run(self):
        """
        Submits a run to the cluster.
        """
        # submit the slurm job
        job_id = auxiliary.submit_slurm_job(self.paths['slurm_script'])

        # check if the job was submitted successfully
        if job_id:
            print(f"""Successfully submitted run {self.run.run_name}
                        with jobid {job_id}""")
            return job_id
        print(f"Some error occurred while submitting run {self.run.run_name}")
        return None
