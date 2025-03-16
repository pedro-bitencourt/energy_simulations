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
from .data_analysis_module import conditional_means
from .constants import (BASE_PATH,
    SOLVER_SLURM_DEFAULT_CONFIG,
    PROCESSING_SLURM_DEFAULT_CONFIG,
    RUN_SLURM_DEFAULT_CONFIG,
    initialize_paths_comparative_statics)
from .utils.load_configs import load_config

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
                 - `conditional_means` [str]: path to the conditional means file.
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

        self.name: str = name
        self.general_parameters: dict = general_parameters
        self.variables: dict = variables

        self._update_general_parameters()
        self.paths: dict = initialize_paths_comparative_statics(
            base_path, name)
        self.solvers: list[Solver] = self._initialize_grid()
        self._validate_parameters()

    def _update_general_parameters(self):
        costs_dictionary = load_costs(self.general_parameters['cost_path'])
        self.general_parameters.update(**costs_dictionary)
        self._set_default_slurm_config()

    def _set_default_slurm_config(self):
        email: dict = self.general_parameters.get("email", "")
        slurm_configs: dict = self.general_parameters.get("slurm", {})

        self.general_parameters["slurm"] = slurm_configs

        run_config: dict = slurm_configs.get("run", RUN_SLURM_DEFAULT_CONFIG)
        self.general_parameters["slurm"]["run"] = self._default_slurm_config(run_config.copy(),
                                                                             RUN_SLURM_DEFAULT_CONFIG, email)

        solver_config: dict = slurm_configs.get(
            "solver", SOLVER_SLURM_DEFAULT_CONFIG)
        self.general_parameters["slurm"]["solver"] = self._default_slurm_config(solver_config.copy(),
                                                                                SOLVER_SLURM_DEFAULT_CONFIG, email)

    def _default_slurm_config(self, slurm_config: dict, default_config: dict,
                              email: str):
        for key, value in default_config.items():
            slurm_config.setdefault(key, value)
        slurm_config.setdefault(
            email, self.general_parameters.get("email"))
        return slurm_config

    def _validate_parameters(self):
        # Check if general parameters contains cost data
        if 'cost_path' not in self.general_parameters:
            raise ValueError(
                "General parameters must contain the cost data path.")

    ############################
    # Initialization methods
    def _initialize_grid(self):
        # Create a Solver object for each combination of exogenous variables
        grid_points = []
        grids = self.variables['exogenous'].values()

        # Currently assuming only one variable
        exogenous_variable_0 = list(self.variables['exogenous'].keys())[0]
        grid_0 = list(grids)[0]['grid']
        # Iterate over exogenous variables
        for value in grid_0:
            exogenous_variables_dict = {
                exogenous_variable_0: {
                    'value': value}
            }
            solver: Solver = self._create_solver(exogenous_variables_dict)
            grid_points.append(solver)
        return grid_points

    def _create_solver(self, exogenous_variables: dict):
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
        investment_problem_0: Solver = self.solvers[0]
        investment_problem_0.prototype()

    def redo_runs(self):
        """
        Deletes and runs again the equilibrium runs.

        WARNING: This method will delete the results of the runs.
        """
        for solver in self.solvers:
            run = solver.last_run()
            logger.info("Deleting and resubmitting run %s", run.name)
            run.delete()
            run.submit(force=True)

    def clear_folders(self):
        for solver in self.solvers:
            solver.clear_runs_folders()

    ############################
    # Submission methods
    def submit_solvers(self):
        """
        Submit the jobs to the cluster.
        """
        for solver in self.solvers:
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

        processing_config = self.general_parameters['slurm'].get('processing',
                                                                 PROCESSING_SLURM_DEFAULT_CONFIG)
        processing_config = {key: processing_config.get(key, default)
                             for key, default in PROCESSING_SLURM_DEFAULT_CONFIG.items()}

        processing_config['email'] = self.general_parameters.get(
            'email', None)
        slurm_path = self.paths['bash'].parent
        header = slurm_header(
            processing_config, f"{self.name}_processing", slurm_path)

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
        self.compute_solver_results()

        self.compute_conditional_means()

    def compute_solver_results(self):
        solver_results_df = solver_results(
            list_of_solvers=self.solvers)
        solver_results_df.to_csv(
            self.paths['solver_results'], index=False)

    def compute_conditional_means(self):
        logger.info("Computing conditional means...")
        conditional_means_df = construct_results(
            list_of_solvers=self.solvers,
            results_function=conditional_means)
        conditional_means_df.to_csv(
            self.paths['conditional_means'], index=False)
        

def extract(list_of_solvers: list[Solver], complete: bool = True, resubmit: bool = False) -> None:
    logger.info("Extracting data from MOP's outputs...")
    for solver in list_of_solvers:
        run: Run = solver.last_run()

        if resubmit and not run.successful(complete=complete):
            logger.error(
                "Run %s not successful. Skipping and resubmitting...", run.name)
            run.submit()
            continue

        run_df = run.full_run_df()

        logger.info(
            "Successfuly extracted data from run %s. Saving to disk...", run.name)
        run_df.to_csv(solver.paths['raw'],
                      index=False)


def profits_results(list_of_solvers: list[Solver]) -> pd.DataFrame:
    rows: list = []
    for solver in list_of_solvers:
        run_df_path = solver.paths['raw']
        rows.append(solver.last_run().get_profits_dict(
            run_df_path=run_df_path))
    results_df: pd.DataFrame = pd.DataFrame(rows)
    return results_df


def solver_results(list_of_solvers: list[Solver]) -> pd.DataFrame:
    rows: list = []
    for solver in list_of_solvers:
        rows.append(solver.solver_results())
    results_df: pd.DataFrame = pd.DataFrame(rows)
    return results_df


def construct_results(list_of_solvers: list[Solver], results_function) -> pd.DataFrame:
    rows = []
    for solver in list_of_solvers:
        results_dict = solver.last_run().results(
            results_function, run_df_path=solver.paths['raw'])
        rows.append(results_dict)
    results_df = pd.DataFrame(rows)
    return results_df


def load_costs(costs_path) -> dict[str, dict]:
    #    if costs_path is None:
    #        costs_path = to do: set a default
    costs_dict: dict[str, dict[str, float | int]
                     ] = load_config(costs_path)
    hourly_costs_dic: dict[str, float] = {
        participant: (costs_dict[participant]['installation'] / costs_dict[participant]
                      ['lifetime'] + costs_dict[participant]['oem']) / 8760
        for participant in costs_dict.keys()
    }

    marginal_costs_dict: dict[str, float | None] = {
        participant: costs_dict[participant].get('marginal_cost')
        for participant in costs_dict.keys()
    }

    result: dict = {'fixed_cost_dictionary': hourly_costs_dic,
                    'marginal_cost_dictionary': marginal_costs_dict}
    return result
