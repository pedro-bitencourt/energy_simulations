"""
Module: comparative_statics_module.py

This module contains the ComparativeStatics class.

Public methods:
- submit: submits the jobs for the runs or investment problems.
- process: processes the results of the runs or investment problems.
- results: gathers the results of the runs or investment problems.

Required modules:
- investment_module: contains the InvestmentProblem class.
- run_module: contains the Run class.
- run_processor_module: contains the RunProcessor class.
- optimization_module: contains the OptimizationPathEntry class.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import json
import pandas as pd

from .auxiliary import submit_slurm_job
from .investment_module import InvestmentProblem
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
        - name: name of the comparative statics exercise.
        - variables: dictionary containing the exogenous and endogenous variables.
        - exogenous_variable_grid: dictionary containing the grids for the exogenous variables.
        - general_parameters: dictionary containing the general parameters, with keys
            o xml_basefile: path to the template xml file.
            o daily: boolean indicating if the runs are daily (True) or weekly (False).
            o name_subfolder: name of the subfolder where the runs are stored.
            o annual_interest_rate: annual interest rate for the investment problems.
            o requested_time_run: requested time for each MOP run to the cluster.
            o requested_time_solver: requested time for each solver job.

    Attributes:
        - name: name of the comparative statics exercise.
        - general_parameters: dictionary containing the general parameters.
        - exogenous_variable: dictionary containing the exogenous variables.
        - endogenous_variables: dictionary containing the endogenous variables.
        - exogenous_variable_grid: dictionary containing the grids for the exogenous variables.
        - paths: dictionary containing the paths for the input, output, and result folders.
        - list_runs: list containing the Run objects.
        - list_investment_problems: list containing the InvestmentProblem objects.
    """

    def __init__(self,
                 name: str,
                 variables: Dict[str, Dict],
                 general_parameters: Dict,
                 base_path: Optional[str] = None):
        """
        Initialize the ComparativeStatics object.
        """
        if base_path is None:
            base_path = str(BASE_PATH)

        self.name: str = name
        self.general_parameters: dict = general_parameters
        self.variables: dict = variables

        # Variables can be exogenous and endogenous
        self.exogenous_variable: dict[str,
                                      dict] = variables.get('exogenous', {})
        self.exogenous_variable_key: str = list(
            self.exogenous_variable.keys())[0]
        self.endogenous_variables: dict[str,
                                        dict] = variables.get('endogenous', {})

        self.exogenous_variable_grid: dict[str, list] = {var_name: variable['grid']
                                                         for var_name, variable in self.exogenous_variable.items()}

        # Initialize relevant paths
        self.paths: dict = initialize_paths_comparative_statics(base_path, name)

        # Create the folders if they do not exist
        for path in self.paths.values():
            if path.is_dir():
                path.mkdir(parents=True, exist_ok=True)

        # Validate the input
        self._validate_input()

        # If there are endogenous variables, we need to handle investment problems first
        if self.endogenous_variables:
            # Create investment problems
            self.list_investment_problems: list[InvestmentProblem] = self._initialize_investment_problems(
            )
            # The runs will be obtained from the investment problems after optimization
            self.list_runs = self.create_runs_from_investment_problems()
        else:
            # Create runs directly
            self.list_runs = self._initialize_runs()

    def prototype(self):
        # Get first investment problem
        investment_problem_0: InvestmentProblem = self.list_investment_problems[0]
        investment_problem_0.prototype()

    def redo_runs(self):
        """
        Deletes and runs again the equilibrium runs.

        WARNING: This method will delete the results of the runs.
        """
        for run in self.list_runs:
            logger.info("Redoing equilibrium run %s", run.name)
            logger.warning("Deleting results of run %s", run.name)
            run.tear_down()
            run.submit()

    def extract_random_variables(self, complete: bool = True) -> None:
        logger.info("Saving random variables DataFrame to results folder...")
        for run in self.list_runs:
            try:
                run_processor = RunProcessor(run, complete=complete)
            except ValueError:
                logger.error(f"Run {run.name} not successful, skipping it")
                continue

            run_df = run_processor.construct_random_variables_df(complete=complete)

            # Copy the random variables to the random_variables folder with the run name
            run_df.to_csv(self.paths['random_variables'] / f"{run.name}.csv",
                          index=False)

    def _validate_input(self) -> None:
        # Check if all variables have a grid
        for variable in self.exogenous_variable:
            if variable not in self.exogenous_variable_grid:
                logging.error(f"Variable {variable} does not have a grid.")
                raise ValueError(f"Variable {variable} does not have a grid.")
        # Check if the grids have the same length
        if len(set([len(grid) for grid in self.exogenous_variable_grid.values()])) != 1:
            logging.error("The grids have different lengths.")
            raise ValueError("The grids have different lengths.")
        # Check if general parameters contain the expected keys
        expected_keys = ['xml_basefile', 'daily', 'name_subfolder']
        if not all(key in self.general_parameters for key in expected_keys):
            logging.error(
                "General parameters do not contain the expected keys.")
            raise ValueError(
                "General parameters do not contain the expected keys.")

    def create_runs_from_investment_problems(self, complete: bool = False):
        """
        Recover the last iteration of each investment problem and create a Run object from it.
        """
        list_runs = []
        for investment_problem in self.list_investment_problems:
            # Create a Run object from the last iteration
            last_run = investment_problem.last_run()
            if last_run.successful(complete=complete):
                list_runs.append(last_run)
            else:
                logger.error("Run %s not successful", last_run.name)
        return list_runs

    def _create_bash(self, lazy):
        # create the data dictionary
        comparative_statics_data = {
            'name': self.name,
            'variables': self.variables,
            'general_parameters': self.general_parameters
        }
        comparative_statics_data = json.dumps(comparative_statics_data)
        print(f"{comparative_statics_data=}")

        if self.general_parameters.get('email', None):
            email_line = f"#SBATCH --mail-user={self.general_parameters['email']}"
        else:
            email_line = ""

        # write the bash script
        with open(self.paths['bash'], 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={PROCESS_TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --job-name={self.name}_processing
#SBATCH --output={self.paths['main']}/{self.name}_processing.out
#SBATCH --error={self.paths['main']}/{self.name}_processing.err
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]
{email_line}

module purge
module load python-miniconda3/4.12.0

python - <<END
import sys
import json
sys.path.append('/projects/p32342/code')
from src.comparative_statics_module import ComparativeStatics

comparative_statics_data = {json.loads(comparative_statics_data, parse_float=float)}
comparative_statics = ComparativeStatics(**comparative_statics_data)
comparative_statics.process({lazy})
END
''')

        return self.paths['bash']

    def submit_processing(self, lazy=True):
        """
        Submit the processing job to the cluster.
        """
        bash_path = self._create_bash(lazy)
        logger.info(f"Submitting processing job for {self.name}")
        job_id = submit_slurm_job(bash_path)
        return job_id

    def _grid_length(self):
        return len(next(iter(self.exogenous_variable_grid.values())))

    def _initialize_runs(self):
        # iterate over the grid
        list_runs: list[Run] = []
        for i in range(self._grid_length()):
            variables = {
                key: {"pattern": self.exogenous_variable[key]["pattern"],
                      "value": self.exogenous_variable_grid[key][i]}
                for key in self.exogenous_variable
            }
            # create a run
            list_runs.append(
                Run(self.paths['main'],
                    self.general_parameters,
                    variables)
            )
        return list_runs

    def _initialize_investment_problems(self):
        # create a list of InvestmentProblem objects
        problems = []
        # iterate over the grid of exogenous variables
        for idx in range(self._grid_length()):
            # get the exogenous variables for the current iteration
            exogenous_variable_temp: dict = {
                variable: {
                    'value': self.exogenous_variable_grid[variable][idx],
                    **var_dict
                }
                for variable, var_dict in self.exogenous_variable.items()
            }
            initial_guesses = {
                variable: var_dict['initial_guess'][idx] if isinstance(
                    var_dict['initial_guess'], list) else var_dict['initial_guess']
                for variable, var_dict in self.endogenous_variables.items()
            }

            endogenous_variables_temp: dict = {
                variable: {
                    'pattern': var_dict['pattern'],
                    'initial_guess': initial_guesses[variable]
                }
                for variable, var_dict in self.endogenous_variables.items()
            }

            # initialize the InvestmentProblem object
            investment_problem = InvestmentProblem(self.paths['main'],
                                                   exogenous_variable_temp,
                                                   endogenous_variables_temp,
                                                   self.general_parameters)

            logger.info('Created investment_problem object for %s.',
                        investment_problem.name)
            problems.append(investment_problem)
        return problems

    def submit(self):
        """
        Submit the jobs to the cluster.
        """
        if self.endogenous_variables:
            # Submit investment problems
            for investment_problem in self.list_investment_problems:
                investment_problem.submit()
                logging.info(
                    f'Submitted job for investment problem %s', investment_problem.name)
        else:
            # Submit runs directly
            for run in self.list_runs:
                job_id = run.submit()
                if job_id is None:
                    logging.error("Failed to submit run %s", run.name)
                else:
                    logging.info("Submitted job for run %s",  run.name)

    def process(self, lazy=True, complete=True):
        """
        """
        # If there are endogenous_variables, first process investment problems and create
        # a list of Run objects
        if self.endogenous_variables:
            # Create runs from investment problems
            self.list_runs = self.create_runs_from_investment_problems()

            # Get the investment results
            investment_results_df = self._investment_results(lazy=lazy)

            # Save to disk
            investment_results_df.to_csv(self.paths['investment_results'], index=False)

        self.paths['random_variables'].mkdir(parents=True, exist_ok=True)

        # Save the random variables df to the random_variables folder
        self.extract_random_variables(complete=complete)

        # Construct the new results dataframe
        conditional_means_df = construct_results(
            self.paths['random_variables'], results_function=conditional_means)

        # Save the results to a .csv file
        conditional_means_df.to_csv(
            self.paths['conditional_means'], index=False)

        # Get daily, weekly, and yearly averages
        # daily_results_df = construct_results(self.paths['random_variables'],
        #                                     results_function=intra_daily_averages)
        # weekly_results_df = construct_results(self.paths['random_variables'],
        #                                      results_function=intra_weekly_averages)
        # yearly_results_df = construct_results(self.paths['random_variables'],
        #                                      results_function=intra_year_averages)

        # Save to disk
        # daily_results_df.to_csv(
        #    self.paths['results'] / 'daily_results.csv', index=False)
        # weekly_results_df.to_csv(
        #    self.paths['results'] / 'weekly_results.csv', index=False)
        # yearly_results_df.to_csv(
        #    self.paths['results'] / 'yearly_results.csv', index=False)

    def _investment_results(self, lazy=True):
        rows: list = []
        for investment_problem in self.list_investment_problems:
            rows.append(investment_problem.investment_results(lazy=lazy))
        results_df: pd.DataFrame = pd.DataFrame(rows)
        return results_df

def construct_results(random_variables_folder: Path, results_function) -> pd.DataFrame:
    runs_list = [run.stem for run in random_variables_folder.iterdir()
                 if run.is_file()]
    # Create a list to store rows
    rows = []

    for run in runs_list:
        # Get the random variables for the current run
        run_random_variables = pd.read_csv(
            random_variables_folder / f'{run}.csv')
        # Get the results to extract for the current run
        results_dict = results_function(run_random_variables)
        # Add run identifier to the results
        results_dict['run'] = run
        # Append the row to our list
        rows.append(results_dict)

    # Create DataFrame from all rows at once
    results_df = pd.DataFrame(rows)

    return results_df
