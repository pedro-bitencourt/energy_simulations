"""
File name: run_module.py
Author: Pedro Bitencourt (Northwestern University)

Description: this file implements the Run class and related methods.
"""

import os
import re
import shutil
from pathlib import Path
import logging
import subprocess
import json
import pandas as pd

from .utils.auxiliary import try_get_file
from .utils.slurm_utils import slurm_header, submit_slurm_job, wait_for_jobs
from .constants import initialize_paths_run, create_run_name
from .data_analysis_module import profits_per_participant, std_variables, full_run_df

logger = logging.getLogger(__name__)

MEMORY_REQUESTED = '5'  # GB


class Run:
    """
    Represents a run and implements some general purpose methods for it.

    Args:
        - parent_folder: Path determining the folder where the run is stored.
        - general_parameters: Dictionary containing general parameters for the run.
            o xml_basefile: Path to the basefile for the .xml file.
            o name_subfolder: Name of the subfolder where the run is stored.
            o requested_time_run: Time requested for the run in the cluster.
        - variables: Dictionary containing the variables for the run. Each entry is 
        a variable, with the key being the variable name and the value being the variable
        value.
        Attributes:
        - name: Name of the run, determined by the folder name.
        - paths: Dictionary containing relevant paths for the run.
    """

    def __init__(self,
                 parent_folder: str,
                 general_parameters: dict,
                 variables: dict[str, float]):
        parent_folder: Path = Path(parent_folder)
        self.variables: dict[str, float] = variables
        self.general_parameters: dict = general_parameters

        self.name: str = create_run_name(variables)
        self.parent_name: str = parent_folder.parts[-1]

        name_subfolder = self.general_parameters.get('name_subfolder',
                                                     'CAD-2024-DIARIA')
        # Initialize relevant paths
        self.paths: dict = initialize_paths_run(
            parent_folder, self.name, name_subfolder)
        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

    def tear_down(self) -> None:
        """
        Deletes the folder and its contents.
        """
        if self.paths['folder'].exists():
            logger.info("Deleting folder %s...", self.paths['folder'])
            try:
                shutil.rmtree(self.paths['folder'])
            except FileNotFoundError:
                logger.warning("Could not delete folder properly")
            logger.info("Deleted folder %s", self.paths['folder'])

    def _get_opt_and_sim_folders(self):
        '''
        Get the opt and sim folders from the subfolder path.
        '''
        opt_sim_paths = {'opt': None, 'sim': None}
        folder_ending = {'opt': "*-OPT", 'sim': "*-SIM"}
        test_files = {'opt': r'tiempos*', 'sim': r'resumIng*'}

        for key in opt_sim_paths:
            candidate_folder_list = list(
                self.paths['subfolder'].glob(folder_ending[key]))
            if candidate_folder_list:
                for folder in candidate_folder_list:
                    correct_folder = try_get_file(
                        folder, test_files[key])
                    if correct_folder:
                        opt_sim_paths[key] = folder
                        continue
        return opt_sim_paths

    def successful(self, complete: bool = False):
        """
        Check if the run was successful by searching for a resumen* file
        in the sim folder.
        """
        # Get opt and sim folders
        self.paths.update(self._get_opt_and_sim_folders())

        # Check if SIM folder exists
        if self.paths.get('sim', False):
            # Check if files exist
            missing_files = self.get_missing_files(self.paths['sim'], complete)
            if missing_files:
                logger.debug(
                    "Run %s is missing files: %s", self.name, missing_files)
                return False
        else:
            logger.debug('No sim folder found for run %s in folder %s',
                         self.name,
                         self.paths['subfolder'])
            return False

        return True

    @staticmethod
    def get_missing_files(sim_path: Path, complete: bool = False):
        """
        Check if the run is missing files.
        """
        files_to_check = [r'resumen*',
                          r'EOLO_eoloDeci/potencias*.xlt',
                          r'FOTOV_solarDeci/potencias*.xlt',
                          r'DEM_demand/potencias*.xlt'
                          ]
        # Add hydro files if complete
        if complete:
            files_to_check.append(r'HID_salto/cota*.xlt')
            files_to_check.append(r'HID_salto/potencias*.xlt')

        missing_files: list[str] = []
        # Check if files exist
        for file_name in files_to_check:
            file_found = try_get_file(sim_path, file_name)
            if not file_found:
                missing_files.append(file_name)
                logger.debug("%s does not contain file %s",
                             sim_path, file_name)
        return missing_files

    def prototype(self):
        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

        xml_path: Path = self.create_xml(
            template_path=self.general_parameters['xml_basefile'],
            name=self.name,
            folder=self.paths['folder'],
            variables=self.variables
        )
        # Create the bash file
        bash_path: Path = self._create_bash(xml_path)

        # Print bash path
        print("Bash path:")
        print(bash_path)
        subprocess.run(['bash', bash_path])

    def submit(self, force: bool = False):
        """
        Submits a run to the cluster.
        """
        # Break if already successful
        if self.successful() and not force:
            logger.info(f"Run {self.name} already successful, skipping.")
            return None

        logger.info("Preparing to submit run %s", self.name)

        logger.warning("Warning: this will overwrite the folder %s",)
        # Tear down the folder
        self.tear_down()

        # Create the directory
        self.paths['folder'].mkdir(parents=True, exist_ok=True)

        # Create the xml file
        xml_path: Path = self.create_xml(
            template_path=self.general_parameters['xml_basefile'],
            name=self.name,
            folder=self.paths['folder'],
            variables=self.variables
        )
        # Create the bash file
        bash_path: Path = self._create_bash(xml_path)

        # Submit the slurm job
        job_id = submit_slurm_job(bash_path, job_name=self.name)

        # Check if the job was submitted successfully
        if job_id:
            logger.info(
                f"Successfully submitted run {self.name} with jobid {job_id}")
            return job_id
        logger.error(f"Some error occurred while submitting run {self.name}")
        return job_id

    ##############################
    # bash creating methods
    def _create_bash(self, xml_path: Path):
        """
        Creates a bash file to be submitted to the cluster.
        """
        bash_path = self.paths['folder'] / f"{self.name}.sh"
        xml_path = os.path.normpath(str(xml_path))
        xml_path = xml_path.replace(os.path.sep, '\\')
        job_name = f"{self.parent_name}_{self.name}"

        slurm_config: dict = self.general_parameters['slurm']['run']
        slurm_path = self.paths['folder']
        header = slurm_header(slurm_config, job_name, slurm_path)

        memory = slurm_config.get('memory', MEMORY_REQUESTED)
        temp_folder_path = f"{self.paths['folder']}/temp"
        temp_folder_path_windows = temp_folder_path.replace('/', '\\')

        with open(bash_path, 'w') as file:
            file.write(f'''{header}
echo "Starting {self.name} at: $(date +'%H:%M:%S')"
export WINEPREFIX=/projects/p32342/software/.wine
mkdir -p {temp_folder_path}
module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
sleep $((RANDOM%60 + 10))
wine "Z:\\projects\\p32342\\software\\Java\\jdk-11.0.22+7\\bin\\java.exe" -Djava.io.tmpdir="Z:\{temp_folder_path_windows}" -Xmx{memory}G -jar MOP_Mingo.JAR "Z:{xml_path}"
''')
        return bash_path

    ##############################
    # xml creating methods
    @staticmethod
    def create_xml(template_path: Path, name: str, folder: Path, variables: dict) -> Path:
        """Creates a .xml file from template with variable substitution."""
        output_path = folder / f"{name}.xml"
        # Add folder path to variables for substitution
        variables = {**variables,
                     'output_folder': str(folder).replace('\\', '\\\\')}
        # Read template and substitute all expressions
        with open(template_path, 'r') as f:
            content = f.read()

        def replace_expr(match):
            expr = match.group(1)
            return str(eval(expr, {}, variables))
        content = re.sub(r'\${([^}]+)}', replace_expr, content)
        # Write output
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path

    def run_df(self, complete: bool = False):
        from .run_processor_module import RunProcessor
        # Initialize the RunProcessor object
        run_processor = RunProcessor(self)

        # Read the run dataframe
        run_df: pd.DataFrame = run_processor.construct_run_df(
            complete=complete)
        return run_df

    def full_run_df(self):
        run_df = self.run_df()
        capacities = self.capacities()
        run_df = full_run_df(run_df, capacities)
        return run_df

    def results(self, results_fun):
        run_df = self.full_run_df()
        results_dict: dict = results_fun(run_df)
        results_dict['name'] = self.name
        results_dict.update(self.variables)
        return results_dict


    def get_profits_data_dict(self, complete: bool = False):
        from .run_processor_module import PARTICIPANTS_LIST_ENDOGENOUS

        participants = PARTICIPANTS_LIST_ENDOGENOUS

        run_df = self.run_df(complete=complete)
        capacities = self.capacities()

        # Compute profits data
        profits_data: dict = profits_per_participant(run_df, capacities, self.general_parameters['cost_path'])

        # Save to disk using json
        self.paths['folder'].joinpath(
            'profits_data.json').write_text(json.dumps(profits_data))

        if complete:
            run_df = full_run_df(run_df, capacities)
            revenues_variables = [f'revenue_{participant}'
                                  for participant in participants]
            std_revenues = std_variables(run_df, revenues_variables)

            # update the profits data with the standard deviations
            profits_data.update(std_revenues)

        logger.debug("profits_data for %s:", self.name)
        logger.debug(profits_data)
        return profits_data

    def capacities(self):
        from .run_processor_module import PARTICIPANTS_LIST_ENDOGENOUS

        participants = PARTICIPANTS_LIST_ENDOGENOUS
        # Create a dictionary with the capacities
        capacities = {participant: self.variables[f"{participant}_capacity"]
                      for participant in participants}
        if 'salto_capacity' in self.variables.keys():
            capacities['salto'] = self.variables['salto_capacity']
        else:
            capacities['salto'] = 1620
        return capacities


    def get_profits(self, complete: bool = False, resubmit: bool = False):
        """
        Computes profits for the specified endogenous variables.

        Returns:
            dict: A dictionary of profits.
        """
        from .run_processor_module import PARTICIPANTS_LIST_ENDOGENOUS
        try:
            profits_dict: dict = self.get_profits_data_dict(complete=complete)
        except FileNotFoundError:
            if resubmit:
                logger.error(
                    "Run %s not successful, resubmitting and returning empty dict",)
                self.submit(force=True)
            else:
                logger.error(
                    "Run %s not successful, returning empty dict",)
            return {}
        profits_dict: dict = {f"{participant}_capacity": profits_dict[f'{participant}_normalized_profits']
                              for participant in PARTICIPANTS_LIST_ENDOGENOUS}
        return profits_dict

def submit_and_wait_for_runs(runs_dict: dict[str, Run],
                    max_attempts: int = 6) -> None:
    '''
    Submits the runs in the runs_dict and waits for them to finish
    '''
    # Submit the runs, give up after max_attempts
    attempts: int = 0
    while attempts < max_attempts:
        job_ids_list: list[str] = []
        for run in runs_dict.values():
            # Check if the run was successful
            if not run.successful():
                logger.warning("Run %s not successful", run.name)
                # If not, submit it and append the job_id to the list
                job_id = run.submit()
                if job_id:
                    job_ids_list.append(job_id)
                if attempts == 0:
                    logger.debug("First attempt for %s", run.name)
                else:
                    logger.warning("RETRYING %s, number of previous attempts = %s", run.name,
                                   attempts)
        if not job_ids_list:
            return None
        logger.info("Waiting for jobs %s", job_ids_list)
        # Wait for the jobs to finish
        wait_for_jobs(job_ids_list)
        attempts += 1

    if attempts == max_attempts:
        logger.critical(
            "Could not successfully finish all runs after %s attempts", max_attempts)
        sys.exit(UNSUCCESSFUL_RUN)
