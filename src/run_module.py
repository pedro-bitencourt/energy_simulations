"""
File name: run_module.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: this file implements the Run class and related methods.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional
import logging
import pandas as pd

import auxiliary


class Run:
    """
    Represents a run and implements some general purpose methods for it.

    Args:
        - folder: Path determining the folder where the run is stored.
        - general_parameters: Dictionary containing general parameters for the run.
            o xml_basefile: Path to the basefile for the .xml file.
            o name_subfolder: Name of the subfolder where the run is stored.
            o requested_time_run: Time requested for the run in the cluster.
        - variables: Dictionary containing the variables for the run. Each entry is 
        a variable, which is a dictionary containing:
            o value: Value of the variable.
            o pattern: Pattern to be substituted in the .xml file.
    Attributes:
        - name: Name of the run, determined by the folder name.
        - paths: Dictionary containing relevant paths for the run.
    """
    def __init__(self,
                 folder: Path,
                 general_parameters: dict,
                 variables: dict[str, dict]):
        self.variables: dict[str, dict] = variables
        self.general_parameters: dict = general_parameters

        self.name: str = Path(folder).name

        # Initialize relevant paths
        self.paths: dict = self._initialize_paths(
            folder, general_parameters)

        # Add paths for results and price distribution files
        self.paths['results_json'] = self.paths['folder'] / 'results.json'
        self.paths['price_distribution'] = self.paths['folder'] / 'price_distribution.csv'

    def _initialize_paths(self, folder: Path, general_parameters: dict):
        """
        Initialize a dictionary with relevant paths for the run.
        """
        def format_windows_path(path_str):
            path_str = str(path_str)
            # Replace forward slashes with backslashes
            windows_path = path_str.replace('/', '\\')
            # Add Z: at the start of the path
            windows_path = 'Z:' + windows_path
            # If the path doesn't end with a backslash, add one
            if not windows_path.endswith('\\'):
                windows_path += '\\'
            return windows_path

        paths = {}
        paths['folder'] = folder
        # Convert the output path to a Windows path, for use in the .xml file
        paths['folder_windows'] = format_windows_path(paths['folder'])

        # Subfolder is PRUEBA or CAD-2024
        subfolder = general_parameters.get('name_subfolder', '')
        paths['subfolder'] = folder / subfolder
        return paths

    def _get_opt_and_sim_folders(self):
        '''
        Get the opt and sim folders from the subfolder path.
        '''
        opt_sim_paths = {'opt': None, 'sim': None}
        folder_ending = {'opt': "*-OPT", 'sim': "*-SIM"}
        test_files = {'opt': r'tiempos*', 'sim': r'resumIng*'}

        for key in opt_sim_paths:
            candidate_folder_list = list(self.paths['subfolder'].glob(folder_ending[key]))
            if candidate_folder_list:
                for folder in candidate_folder_list:
                    correct_folder = auxiliary.try_get_file(
                        folder, test_files[key])
                    if correct_folder:
                        opt_sim_paths[key] = folder
                        continue
            else:
                logging.error('No %s folder found for run %s in folder %s',
                              key, self.name, self.paths['subfolder'])
        return opt_sim_paths

    def successful(self):
        """
        Check if the run was successful by searching for a resumen* file
        in the sim folder.
        """
        # Get opt and sim folders
        self.paths.update(self._get_opt_and_sim_folders())

        # Check if SIM folder exists
        sim_folder = self.paths.get('sim', False)
        if sim_folder:
            # Check if the resumen file exists
            resumen_file = auxiliary.try_get_file(sim_folder, r'resumen*')
            if resumen_file:
                return True
            logging.error('No resumen file found for run %s in folder %s',
                          self.name,
                          self.paths['subfolder'])
        else:
            logging.error('No sim folder found for run %s in folder %s',
                          self.name,
                          self.paths['subfolder'])
        return False

    def submit(self):
        """
        Submits a run to the cluster.
        """
        # Break if already successful
        if self.successful():
            print(f"Run {self.name} already successful, skipping.")
            return True

        print(f"""Preparing to submit run {self.name},
                    with successful status {self.successful()}""")

        # Create the xml file
        xml_path: Path = self._create_xml()
        # Create the bash file
        bash_path: Path = self._create_bash(xml_path)

        # Submit the slurm job
        job_id = auxiliary.submit_slurm_job(bash_path)

        # Check if the job was submitted successfully
        if job_id:
            logging.info(f"Successfully submitted run {self.name} with jobid {job_id}")
            return job_id
        logging.error(f"Some error occurred while submitting run {self.name}")
        return job_id

    def _create_xml(self):
        """
        Creates a .xml file from the .xml template.
        """

        xml_path: Path = self.paths['folder'] / f"{self.name}.xml"
        # Open the experiment's .xml basefile
        with open(self.general_parameters['xml_basefile'], 'r') as file:
            content = file.read()

            def escape_for_regex(path):
                return path.replace('\\', '\\\\')

            # Change output path
            content = re.sub(
                r'<rutaSalidas>.*?</rutaSalidas>',
                f'<rutaSalidas>{escape_for_regex(self.paths["folder_windows"])}</rutaSalidas>',
                content
            )

            # Substitute variables values
            content = self._substitute_variables(content)

        # Write the new settings to the .xml file path
        with open(xml_path, 'w') as file:
            file.write(content)

        return xml_path

    def _substitute_variables(self, content):
        """
        Substitutes patterns in the content with the variable values.
        """
        DEGREES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        def substitute_pattern(content, pattern, value):
            """
            Substitutes a pattern in the content with a given value.
            """
            def replace_func(match):
                if '*' in match.group():
                    # Get the captured multiplier
                    multiplier = float(match.group(2))
                    # Changed from int() to float()
                    new_value = float(value * multiplier)
                    return str(new_value)
                return str(value)

            # First try to match pattern with multiplier
            content = re.sub(
                f'({pattern})(?:\*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))', replace_func, content)

            # Then match pattern without multiplier
            content = re.sub(
                f'({pattern})<', replace_func, content)

            return content

        for variable in self.variables.values():
            for degree in DEGREES:
                pattern = variable['pattern'] + f"_{degree}"
                value = variable['value']**degree
                content = substitute_pattern(content, pattern, value)
            content = substitute_pattern(
                content, variable['pattern'], variable['value'])
        return content

    def _create_bash(self, xml_path: Path):
        """
        Creates a bash file to be submitted to the cluster.
        """
        bash_path = self.paths['folder'] / f"{self.name}.sh"
        xml_path = os.path.normpath(str(xml_path))
        xml_path = xml_path.replace(os.path.sep, '\\')

        requested_time = self.general_parameters.get('requested_time_run', '24:00:00')

        with open(bash_path, 'w') as file:
            file.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time}
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=5G 
#SBATCH --job-name={self.name}
#SBATCH --output={self.paths['folder']}/{self.name}.out
#SBATCH --error={self.paths['folder']}/{self.name}.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
sleep $((RANDOM%60 + 10)) 
wine "Z:\\projects\\p32342\\software\\Java\\jdk-11.0.22+7\\bin\\java.exe" -Xmx5G -jar MOP_Mingo.JAR "Z:{xml_path}"
''')
        return bash_path

    def load_results(self) -> Optional[dict]:
        """
        Loads the results from the results.json file.
        """
        if self.paths['results_json'].exists():
            with open(self.paths['results_json'], 'r') as file:
                results = json.load(file)
            return results
        logging.error(f"Results file {self.paths['results_json']} does not exist for run {self.name}")
        return None

    def load_price_distribution(self) -> Optional[pd.DataFrame]:
        """
        Loads the price distribution from the CSV file.
        """
        if self.paths['price_distribution'].exists():
            price_distribution: pd.DataFrame = pd.read_csv(self.paths['price_distribution'])
            # Convert 'hour_of_week_bin' to just the start hour
            price_distribution['hour_of_week_bin'] = (
                price_distribution['hour_of_week_bin'].str.extract(r'(\d+)'))
            # Pivot the table to have hours as columns
            wide_df = price_distribution.pivot_table(
                values='price_avg',
                index=None,
                columns='hour_of_week_bin'
            )
            # Convert all column names to integers and sort
            wide_df.columns = wide_df.columns.astype(int)
            wide_df = wide_df.sort_index(axis=1).reset_index(drop=True)
            return price_distribution
        logging.error(f"Price distribution file {self.paths['price_distribution']} does not exist for run {self.name}")
        return None
