"""
File name: run_module.py
Author: Pedro Bitencourt (Northwestern University)

Description: this file implements the Run class and related methods.
"""

import os
import re
import sys
import shutil
from pathlib import Path
import logging
import subprocess
import json
import pandas as pd

from .utils.auxiliary import try_get_file
from .utils.slurm_utils import slurm_header, submit_slurm_job, wait_for_jobs
from .constants import RUN_SLURM_DEFAULT_CONFIG, initialize_paths_run, create_run_name
from .data_analysis_module import profits_per_participant, full_run_df

logger = logging.getLogger(__name__)

MEMORY_REQUESTED = '5'  # GB
PARTICIPANTS_DEFAULT: list = ['wind', 'solar',
                              'thermal', 'salto']
PARTICIPANTS_ENDOGENOUS_DEFAULT: list = ['wind', 'solar', 'thermal']


class Run:
    """
    Represents a run and implements some general purpose methods for it.

    Args:
        - parent_folder: Path determining the folder where the run is stored.
        - general_parameters: Dictionary containing general parameters for the run,
            keys as ComparativeStatic.
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
        self.folder: Path = self.paths['folder']

        self.log_run()

    def capacities(self):
        participants: list[str] = self.general_parameters.get(
            'endogenous_participants', PARTICIPANTS_ENDOGENOUS_DEFAULT
        )
        # Create a dictionary with the capacities
        capacities = {participant: self.variables[f"{participant}_capacity"]
                      for participant in participants}
        if 'salto_capacity' in self.variables.keys():
            capacities['salto'] = self.variables['salto_capacity']
        else:
            capacities['salto'] = 810
        return capacities

    def log_run(self):
        logger.debug("Run %s", self.name)
        logger.debug("=" * 40)
        logger.debug(f"Parent Folder: {self.folder}")
        logger.debug(f"Parent Name: {self.parent_name}")
        logger.debug(f"Run Name: {self.name}")
        logger.debug(f"General Parameters: {self.general_parameters}")
        logger.debug(f"Variables:")
        for key, value in self.variables.items():
            logger.debug(f"  {key}: {value}")
        logger.debug(f"Paths:")
        for key, path in self.paths.items():
            logger.debug(f"  {key}: {path}")
        logger.debug("=" * 40)

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
            variables=self.variables,
            marginal_cost=self.general_parameters['marginal_cost_dictionary']
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
            variables=self.variables,
            marginal_cost=self.general_parameters['marginal_cost_dictionary']
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

        slurm_config: dict = self.general_parameters.get("slurm", {})
        run_config = slurm_config.get("run", RUN_SLURM_DEFAULT_CONFIG)
        run_config = {
            key: run_config.get(key, value)
            for key, value in RUN_SLURM_DEFAULT_CONFIG.items()
        }
        run_config['email'] = self.general_parameters.get(
            'email', None)

        slurm_path = self.paths['folder']
        header = slurm_header(run_config, job_name, slurm_path)

        memory = run_config.get('memory', MEMORY_REQUESTED)
        temp_folder_path = f"/tmp/pdm6134/{self.name}"
        temp_folder_path_windows = temp_folder_path.replace('/', '\\')
        wine_path = self.general_parameters.get(
            'wine_path', '/projects/p32342/software/.wine')

        with open(bash_path, 'w') as file:
            file.write(f'''{header}
echo "Starting {self.name} at: $(date +'%H:%M:%S')"
export WINEPREFIX={wine_path}
mkdir -p {temp_folder_path}
module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
sleep $((RANDOM%60 + 10))
wine "Z:\\projects\\p32342\\software\\Java\\jdk-11.0.22+7\\bin\\java.exe" -Djava.io.tmpdir="Z:{temp_folder_path_windows}" -Xmx{memory}G -jar MOP_Mingo.JAR "Z:{xml_path}"
rm -r {temp_folder_path}
''')
        return bash_path

    ##############################
    # xml creating methods
    @staticmethod
    def create_xml(template_path: Path, name: str, folder: Path, variables: dict,
                   marginal_cost: dict) -> Path:
        """Creates a .xml file from template with variable substitution."""
        output_path = folder / f"{name}.xml"
        marginal_cost_variables = {
            f'marginal_cost_{participant}': marginal_cost[participant]
            for participant in marginal_cost.keys()
        }
        # Add folder path to variables for substitution
        variables = {**variables,
                     'output_folder': str(folder).replace('\\', '\\\\'),
                     **marginal_cost_variables}

        # Read template
        with open(template_path, 'r') as f:
            content = f.read()

        def replace_expr(match):
            # Replaces a ${expression} in the template by its values, using
            # the input from self.variables
            expr = match.group(1)
            return str(eval(expr, {}, variables))

        content = re.sub(r'\${([^}]+)}', replace_expr, content)
        # Write output
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path

    ##############################
    # Processing methods
    def run_df(self, complete: bool = False):
        from .run_processor_module import RunProcessor
        # Initialize the RunProcessor object
        run_processor = RunProcessor(self)

        # Extract the run dataframe
        run_df: pd.DataFrame = run_processor.construct_run_df(
            complete=complete)
        return run_df

    def full_run_df(self, run_df=None):
        if run_df is None:
            run_df = self.run_df(complete=True)
        capacities = self.capacities()
        participants = self.general_parameters.get(
            'participants', PARTICIPANTS_DEFAULT)
        run_df = full_run_df(run_df, participants)
        return run_df

    def results(self, results_fun, run_df_path=None, complete=True):
        """
        Computes results according to 'results_fun' on the run_df.
        Args:
            - results_fun: should be a function that takes full_run_df and the
            capacities dictionary as input and outputs a dictionary.
            - run_df_path: (optional) should be the path to the run_df.
            - complete: (optional) should describe if the full run_df is needed.
        """
        if run_df_path is not None:
            run_df: pd.DataFrame = pd.read_csv(run_df_path)
            run_df = self.full_run_df(run_df)
            if complete:
                run_df: pd.DataFrame = self.full_run_df(run_df)
        else:
            run_df: pd.DataFrame = self.run_df(complete=complete)

        capacities = self.capacities()
        results_dict: dict = results_fun(run_df, capacities)
        results_dict['name'] = self.name
        results_dict.update(self.variables)
        return results_dict

    # Refactor
    def get_profits_dict(self, complete: bool = False, run_df_path=None):
        """
        To do: refactor this so it uses the results() method.
        """
        participants = self.general_parameters.get(
            'endogenous_participants', PARTICIPANTS_ENDOGENOUS_DEFAULT)

        if run_df_path is not None:
            run_df: pd.DataFrame = pd.read_csv(run_df_path)
        else:
            run_df = self.run_df(complete=complete)
        capacities = self.capacities()

        profits_dict: dict = profits_per_participant(
            run_df, capacities, self.general_parameters['fixed_cost_dictionary'])

        # Save to disk using json
        self.paths['folder'].joinpath(
            'profits_data.json').write_text(json.dumps(profits_dict))

        if complete:
            # TODO : CHECK this part
            run_df = full_run_df(run_df, participants)
            revenues_variables = [f'revenue_{participant}'
                                  for participant in participants]

        logger.debug("profits_data for %s:", self.name)
        logger.debug(profits_dict)
        return profits_dict

    # Refactor
    def get_profits(self, complete: bool = False, resubmit: bool = False):
        """
        Computes profits for the specified endogenous variables.

        Returns:
            dict: A dictionary of profits.
        """
        participants = self.general_parameters.get(
            'endogenous_participants', PARTICIPANTS_ENDOGENOUS_DEFAULT)
        try:
            profits_dict: dict = self.get_profits_dict(complete=complete)
        except FileNotFoundError as file_error:
            logger.error(
                "Run %s not successful, resubmitting and returning empty dict", self.name)
            logger.error("File error: %s", file_error)
            if resubmit:
                self.submit(force=True)
            return {}
        profits_dict: dict = {f"{participant}_capacity": profits_dict[f'{participant}_normalized_profits']
                              for participant in participants}
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
        sys.exit()
