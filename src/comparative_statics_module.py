"""
Module: comparative_statics_module.py
Author: Pedro Bitencourt (Northwestern University)
Last modified: 12.17.24


Description:
    This module contains the ComparativeStatics class, which is the main class used in this project.
    This class models a comparative statics exercise to be executed and processed, using both the
    other scripts in this folder as the Modelo de Operaciones Padron (MOP), which implements a
    solver for the problem of economic dispatch of energy for a given configuration of the energy
    system.

Classes:
    - ComparativeStatistics
        - Attributes:
            - `name` [str]: name for the exercise, to be used for the creation of folders and files
            - `general_parameters` [dict]: dictionary containing general options for the program, with
            keys:
                o xml_basefile [str]: path to the template xml file.
                o daily [bool]: boolean indicating if the runs are daily (True) or weekly (False).
                o annual_interest_rate [float]: annual interest rate for the investment problems.
                o slurm [dict]: dictionary containing options for slurm, keys:
                    - `run`:
                    - `solver`:
                    Each of these contains the options:
                        - `email` [str]
                        - `mail-type` [str]
                        - `time` [float]: in hours
                        - `memory` [str]: in GB
                o `solver` [dict]: dictionary containing options for the solver
        - Methods:
            - `submit_solvers`: submits all Solvers for the exercise.
            - `submit_processing`: submits a processing job for the exercise.
"""

import logging
from typing import Optional, Dict
import json
import pandas as pd

from .utils.slurm_utils import submit_slurm_job, slurm_header
from .solver_module import Solver
from .run_module import Run
from .run_processor_module import RunProcessor
from .data_analysis_module import conditional_means
from .constants import BASE_PATH, initialize_paths_comparative_statics

# Get logger for current module
logger = logging.getLogger(__name__)

PROCESS_TIME = '05:00:00'


class ComparativeStatics:
    """
    Represents a comparative statics object.
    Arguments:
        - name [str]: name of the comparative statics exercise.
        - variables: [dict[str, dict]] dictionary containing the exogenous and endogenous variables.
        - general_parameters [dict[str, dict]]: dictionary containing the general parameters. Keys:
            o xml_basefile [str]: path to the template xml file.
            o daily [bool]: boolean indicating if the runs are daily (True) or weekly (False).
            o annual_interest_rate [float]: annual interest rate for the investment problems.
            o slurm [dict]: dictionary containing options for slurm, keys:
                - `run`:
                - `solver`:
                Each of these contains the options:
                    - `email`:
                    - `mail-type`:
                    - `time`:
            o `solver` [dict]: dictionary containing options for the solver

    Attributes:
        - name: name of the comparative statics exercise.
        - general_parameters: dictionary containing the general parameters.
        - paths: dictionary containing the paths for the input, output, and result folders.
        - grid_points: list containing the Solver objects.
    """

    def __init__(self,
                 name: str,
                 variables: Dict[str, Dict],
                 general_parameters: Dict,
                 base_path: Optional[str] = None):
        """
        Initialize the ComparativeStatics object.
        """
        logger.info("Initializing the ComparativeStatics object.")
        if base_path is None:
            base_path = str(BASE_PATH)

        self.name: str = name
        self.general_parameters: dict = general_parameters
        self.variables: dict = variables
        self.paths: dict = initialize_paths_comparative_statics(
            base_path, name)
        self.grid_points: list[Solver] = self.initialize_grid()
        self.validate_parameters()

    def validate_parameters(self):
        # Check if general parameters contains cost data
        if 'cost_path' not in self.general_parameters:
            raise ValueError(
                "General parameters must contain the cost data path.")

    ############################
    # Initialization methods
    def initialize_grid(self):
        # Create a Solver object for each combination of exogenous variables
        grid_points = []
        grids = self.variables['exogenous'].values()

        exogenous_variable_0 = list(self.variables['exogenous'].keys())[0]
        grid_0 = list(grids)[0]['grid']
        # Iterate over exogenous variables
        for value in grid_0:
            exogenous_variables_dict = {
                exogenous_variable_0: {
                    'value': value}
            }
            solver: Solver = self.create_solver(exogenous_variables_dict)
            grid_points.append(solver)
        return grid_points

    def create_solver(self, exogenous_variables: dict):
        """
        Create an investment problem for the given exogenous variables.
        """
        solver = Solver(self.paths['main'],
                        exogenous_variables,
                        self.variables['endogenous'],
                        self.general_parameters)
        return solver

    ############################
    # Utility methods
    def prototype(self):
        # Get first investment problem
        investment_problem_0: Solver = self.grid_points[0]
        investment_problem_0.prototype()

    def redo_runs(self):
        """
        Deletes and runs again the equilibrium runs.

        WARNING: This method will delete the results of the runs.
        """
        for solver in self.grid_points:
            run = solver.last_run()
            logger.info("Redoing equilibrium run %s", run.name)
            run.submit(force=True)

    def clear_folders(self):
        for solver in self.grid_points:
            solver.clear_runs_folders()

    ############################
    # Submission methods
    def submit_solvers(self):
        """
        Submit the jobs to the cluster.
        """
        for solver in self.grid_points:
            solver.submit()
            logging.info(
                'Submitted job for investment problem %s', solver.name)

    def _create_bash(self):
        comparative_statics_data = {
            'name': self.name,
            'variables': self.variables,
            'general_parameters': self.general_parameters
        }
        comparative_statics_data = json.dumps(comparative_statics_data)

        slurm_config = self.general_parameters['slurm']['processing']
        slurm_path = self.paths['bash'].parent
        header = slurm_header(
            slurm_config, f"{self.name}_processing", slurm_path)

        with open(self.paths['bash'], 'w') as f:
            f.write(f'''{header}
module purge
module load python-miniconda3/4.12.0

python - <<END
import sys
import json
import logging
sys.path.append('/projects/p32342/code')
from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

setup_logging(level = logging.DEBUG)

comparative_statics_data = {json.loads(comparative_statics_data, parse_float=float)}
comparative_statics = ComparativeStatics(**comparative_statics_data)
comparative_statics.process()
comparative_statics.clear_folders()
END
''')

        return self.paths['bash']

    def submit_processing(self):
        """
        Submit the processing job to the cluster.
        """
        bash_path = self._create_bash()
        logger.info(f"Submitting processing job for {self.name}")
        job_id = submit_slurm_job(
            bash_path, job_name=f"{self.name}_processing")
        return job_id

    ############################
    # Processing methods
    def process(self, complete=True):
        self.extract_runs_dataframes(complete=complete)
        solver_results_df = self.solver_results()
        solver_results_df.to_csv(
            self.paths['solver_results'], index=False)

        conditional_means_df = self.construct_results(
            results_function=conditional_means)
        conditional_means_df.to_csv(
            self.paths['conditional_means'], index=False)

    def extract_runs_dataframes(self, complete: bool = True) -> None:
        logger.info("Extracting data from MOP's outputs...")
        for solver in self.grid_points:
            run: Run = solver.last_run()
            try:
                run_processor: RunProcessor = RunProcessor(
                    run, complete=complete)
            except FileNotFoundError:
                logger.error(
                    "Run %s not successful. Skipping and resubmitting...", run.name)
                run.submit()
                continue

            run_df = run_processor.construct_run_df(
                complete=complete)

            logger.info(
                "Successfuly extracted data from run %s. Saving to disk...", run.name)
            run_df.to_csv(self.paths['raw'] / f"{solver.name}.csv",
                          index=False)

    def solver_results(self):
        rows: list = []
        for solver in self.grid_points:
            rows.append(solver.solver_results())
        results_df: pd.DataFrame = pd.DataFrame(rows)
        return results_df

    def construct_results(self, results_function) -> pd.DataFrame:
        # Create a list to store rows
        rows = []
        for point in self.grid_points:
            results_dict = point.last_run().results(results_function)
            # Append the row to our list
            rows.append(results_dict)
        results_df = pd.DataFrame(rows)
        return results_df
