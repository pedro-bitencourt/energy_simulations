"""
Module: comparative_statics_module.py
Author: Pedro Bitencourt (Northwestern University)
Last modified: 02.24.25


Description:
    This module contains the ComparativeStatics class, which is the main class used in this project.
    This class models a comparative statics exercise to be executed and processed, using both the
    other scripts in this folder as the Modelo de Operaciones Padron (MOP), which implements a
    solver for the problem of economic dispatch of 

Classes:
    - ComparativeStatistics
        - Attributes:
            - `name` [str]: name for the exercise, to be used for the creation of folders and files
            - `general_parameters` [dict]: dictionary containing general options for the program,
            with keys:
                - cost_path [str]: path to the cost data file.
                - xml_basefile [str]: path to the template xml file.
                - daily [bool]: boolean indicating if the runs are daily (True) or weekly (False).
                - annual_interest_rate [float]: annual interest rate for the investment problems.
                - `solver` [dict]: dictionary containing options for the solver
                - slurm [dict]: dictionary containing options for slurm, keys:
                    - `run`: (extra key: 'wine_path', see Run.submit())
                    - `solver`:
                    - `processing`:
                    Each of these contains the options:
                        - `email` [str]
                        - `mail-type` [str]
                        - `time` [float]: in hours
                        - `memory` [str]: in GB
        - Methods:
            - `submit_solvers`: submits all Solvers for the exercise.
            - `submit_processing`: submits a processing job for the exercise.
            - `clear_folders`: deletes the folder for all non-equilibrium runs.
"""
import logging
from typing import Optional, Dict
import json
import pandas as pd

from .utils.slurm_utils import submit_slurm_job, slurm_header
from .solver_module import Solver
from .run_module import Run
from .constants import (BASE_PATH,
                        initialize_paths_comparative_statics)
from .utils.load_configs import load_costs

# Get logger for current module
logger = logging.getLogger(__name__)


class ComparativeStatics:
    """
        - Inputs (which are also attributes):
            - `name` [str]: name for the exercise, to be used for the creation of folders and files
            - `general_parameters` [dict]: dictionary containing general options for the program,
            with keys:
                - cost_path [str]: path to the cost data file.
                - xml_basefile [str]: path to the template xml file.
                - daily [bool]: boolean indicating if the runs are daily (True) or weekly (False).
                - annual_interest_rate [float]: annual interest rate for the investment problems.
                - `solver` [dict]: dictionary containing options for the solver
                - slurm [dict]: dictionary containing options for slurm, keys:
                    - `run`: (extra key: 'wine_path', see Run.submit())
                    - `solver`:
                    - `processing`:
                    Each of these contains the options:
                        - `email` [str]
                        - `mail-type` [str]
                        - `time` [float]: in hours
                        - `memory` [str]: in GB
            - `variables` [dict]: dictionary containing the variables for the exercise, with keys:
                - `exogenous` [dict]: dictionary containing the exogenous variables, with keys:
                    - `name` [str]: dictionary containing the name of the variable, with keys:
                        - `grid` [list]: list of values for the variable.
                        - `unit` [str]: unit for the variable.
                - `endogenous` [dict]: dictionary containing the endogenous variables, with keys:
                    - `name` [str]: dictionary containing the name of the variable, with keys:
                        - `unit` [str]: unit for the variable.
                - `base_path` [str]: path to the base folder for the exercise.
        - Attributes:
             - `paths` [dict]: dictionary containing the paths for the exercise, with keys:
                 - `main` [str]: path to the main folder for the exercise.
                 - `bash` [str]: path to the bash file for the exercise.
                 - `solver_results` [str]: path to the solver results file.
             - `list_of_solvers` [list]: list of Solver objects for the exercise.


       - Methods:
           - `submit_solvers`: submits all Solvers for the exercise.
           - `submit_processing`: submits a processing job for the exercise.
           - `clear_folders`: deletes the folder for all non-equilibrium runs.
    """

    def __init__(self,
                 name: str,
                 variables: Dict[str, Dict],
                 general_parameters: Dict,
                 base_path: Optional[str] = None):
        logger.info("Initializing the ComparativeStatics object.")
        if base_path is None:
            base_path = str(BASE_PATH)
            print(base_path)

        self.name: str = name
        self.general_parameters: dict = general_parameters
        self.variables: dict = variables

        self._update_general_parameters()
        self.paths: dict = initialize_paths_comparative_statics(
            base_path, name)
        self.solvers: list[Solver] = self._initialize_solvers()
        self._validate_parameters()

    def _update_general_parameters(self):
        costs_dictionary = load_costs(self.general_parameters['cost_path'])
        self.general_parameters.update({"cost_parameters": costs_dictionary})

    def _validate_parameters(self):
        # Check if general parameters contains cost data
        if 'cost_path' not in self.general_parameters:
            raise ValueError(
                "General parameters must contain the cost data path.")

    ############################
    # Initialization methods
    def _initialize_solvers(self):
        """
        Initialize the solvers for the exercise. Currently only implemented for one exogenous
        variable.
        """
        grid_points = []
        exogenous_variable_name, grid = list(
            self.variables['exogenous'].items())[0]
        for value in grid['grid']:
            grid_points.append(self._construct_solver(
                exogenous_variable_name, value))
        return grid_points

    def _construct_solver(self, exogenous_variable: str, value: float) -> Solver:
        exogenous_variables_dict = {
            exogenous_variable: {
                'value': value}
        }
        solver = Solver(self.paths['main'],
                        exogenous_variables_dict,
                        self.variables['endogenous'],
                        self.general_parameters)
        return solver

    ############################
    # Utility methods
    def prototype(self):
        investment_problem_0: Solver = self.solvers[0]
        investment_problem_0.prototype()

    def broadcast(self, method_name: str, *args, **kwargs):
        for solver in self.solvers:
            getattr(solver, method_name)(*args, **kwargs)

    def submit_solvers(self):
        self.broadcast('submit')

    def redo_runs(self, delete: bool = False):
        self.broadcast('redo', delete=delete)

    def clear_folders(self):
        self.broadcast('clear_runs_folders')

    def serialize(self):
        return json.dumps({
            'name': self.name,
            'variables': self.variables,
            'general_parameters': self.general_parameters
        })

    ############################
    def _create_bash(self):
        comparative_statics_data = self.serialize()

        header = slurm_header(
            slurm_configs=self.general_parameters['slurm'],
            job_name=f"{self.name}_processing",
            job_type="processing",
            slurm_path=self.paths['bash'],
            email=self.general_parameters['slurm'].get('email')
        )
        with open(self.paths['bash'], 'w') as f:
            f.write(f'''{header}
module purge
module load python-miniconda3/4.12.0

python - <<END
import sys
import json
sys.path.append('/projects/p32342/code')
from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

setup_logging(level = "info")

comparative_statics_data = {json.loads(
    comparative_statics_data, parse_float=float)}
comparative_statics = ComparativeStatics(**comparative_statics_data)
comparative_statics.process()
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
    def process(self, complete=True, resubmit=True):
        print("Creating raw directory...")
        self.paths['raw'].mkdir(exist_ok=True, parents=True)
        extract(list_of_solvers=self.solvers,
                complete=complete, resubmit=resubmit)
        solver_results(list_of_solvers=self.solvers).to_csv(
            self.paths['solver_results'], index=False)


def extract(list_of_solvers: list[Solver], complete: bool = True, resubmit: bool = False) -> None:
    logger.info("Extracting data from MOP's outputs...")
    for solver in list_of_solvers:
        run: Run = solver.last_run()

        if resubmit and not run.successful(complete=complete):
            logger.error(
                "Run %s not successful. Skipping and resubmitting...", run.name)
            run.submit()
            continue

        run_df = run.run_df(complete=complete)

        logger.info(
            "Successfuly extracted data from run %s. Saving to disk...", run.name)
        run_df.to_csv(solver.paths['raw'],
                      index=False)


def solver_results(list_of_solvers: list[Solver]) -> pd.DataFrame:
    rows: list = [solver.solver_results() for solver in list_of_solvers]
    return pd.DataFrame(rows)
