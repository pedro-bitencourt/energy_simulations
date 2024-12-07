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

from src.auxiliary import submit_slurm_job

from .investment_module import InvestmentProblem
from .run_module import Run
from .run_processor_module import RunProcessor
from .optimization_module import OptimizationPathEntry, get_last_successful_iteration
from .auxiliary import submit_slurm_job

from src.constants import BASE_PATH

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
        - list_simulations: list containing the Run objects.
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
        self.paths: dict = self._initialize_paths(base_path)

        # Create the folders if they do not exist
        for path in self.paths.values():
            if path.is_dir():
                path.mkdir(parents=True, exist_ok=True)

        # Validate the input
        self._validate_input()

        # Initialize the list of simulations
        self.list_simulations: list[Run] = []

        # If there are endogenous variables, we need to handle investment problems first
        if self.endogenous_variables:
            # Create investment problems
            self.list_investment_problems: list[InvestmentProblem] = self._initialize_investment_problems(
            )
            # The runs will be obtained from the investment problems after optimization
            self.list_simulations = self.create_runs_from_investment_problems()
        else:
            # Create runs directly
            self.list_simulations = self._initialize_runs()

    def prototype(self):
        # Get first investment problem
        investment_problem_0: InvestmentProblem = self.list_investment_problems[0]
        investment_problem_0.prototype()

    def redo_equilibrium_runs(self):
        """
        Deletes and runs again the equilibrium runs.
        """
        for run in self.list_simulations:
            logger.info("Redoing equilibrium run %s", run.name)
            run.tear_down()
            run.submit()

    def get_random_variables_df(self, lazy=True) -> pd.DataFrame:
        logger.info("Saving random variables DataFrame to results folder...")
        df_list: list[pd.DataFrame] = []
        for run in self.list_simulations:
            try:
                run_processor = RunProcessor(run, complete=True)
            except ValueError:
                logger.error(f"Run {run.name} not successful, skipping it")
                continue

            run_df = run_processor.get_random_variables_df(lazy)

            # Copy the random variables to the random_variables folder with the run name
            run_df.to_csv(self.paths['random_variables'] / f"{run.name}.csv",
                          index=False)

    def _validate_input(self):
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

    def create_runs_from_investment_problems(self, check_convergence:
                                             bool = False):
        """
        Recover the last iteration of each investment problem and create a Run object from it.
        """
        list_simulations = []
        for investment_problem in self.list_investment_problems:
            # Load the last iteration as the Run
            last_iteration = get_last_successful_iteration(
                investment_problem.optimization_trajectory)
            if last_iteration is None:
                logging.warning(
                    f"No successful iteration found for investment problem {investment_problem.name}")
                continue
            if check_convergence and not last_iteration.check_convergence():
                logging.error(
                    f"Investment problem {investment_problem.name} did not converge, skipping")
                continue
            else:
                logging.info(
                    f"Investment problem {investment_problem.name} converged")

            # Create a Run object from the last iteration
            equilibrium_run = investment_problem.create_run(
                last_iteration.current_investment)

            if last_iteration.profits is not None:
                # Clear the folder for the other runs
                logger.info("Deleting run folders for investment problem %s with equilibrium run %s",
                            investment_problem.name, equilibrium_run.name)
                investment_problem._clear_runs_folders(equilibrium_run.name)
            list_simulations.append(equilibrium_run)
        return list_simulations

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

    def _initialize_paths(self, base_path: str) -> dict:
        paths = {}
        paths['main'] = Path(f"{base_path}/comparative_statics/{self.name}")
        paths['results'] = Path(f"{base_path}/results/{self.name}")
        paths['random_variables'] = Path(
            f"{base_path}/results/{self.name}/random_variables")
        paths['bash'] = Path(
            f"{base_path}/comparative_statics/{self.name}/process.sh")
        return paths

    def _initialize_runs(self):
        # iterate over the grid
        list_simulations: list[Run] = []
        for i in range(self._grid_length()):
            variables = {
                key: {"pattern": self.exogenous_variable[key]["pattern"],
                      "value": self.exogenous_variable_grid[key][i]}
                for key in self.exogenous_variable
            }
            # create a run
            list_simulations.append(
                Run(self.paths['main'],
                    self.general_parameters,
                    variables)
            )
        return list_simulations

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
            for inv_prob in self.list_investment_problems:
                if inv_prob.check_convergence():
                    logging.info("Investment problem %s reached convergence",
                                 inv_prob.name)
                    continue
                else:
                    inv_prob.submit()
                    logging.info(
                        f'Submitted job for investment problem %s', inv_prob.name)
        else:
            # Submit runs directly
            for run in self.list_simulations:
                job_id = run.submit()
                if job_id is None:
                    logging.error("Failed to submit run %s", run.name)
                else:
                    logging.info("Submitted job for run %s",  run.name)

    def process(self, lazy=True):
        """
        """
        # If there are endogenous_variables, first process investment problems and create
        # a list of Run objects
        if self.endogenous_variables:
            # Create runs from investment problems
            self.list_simulations = self.create_runs_from_investment_problems()

            # Get the investment results
            investment_results_df = self._investment_results(lazy)

            # Save to disk
            investment_results_df.to_csv(
                self.paths['results'] / 'investment_results.csv', index=False)

        self.paths['random_variables'].mkdir(parents=True, exist_ok=True)

        # Save the random variables df to the random_variables folder
        self.get_random_variables_df(lazy)

        # Construct the new results dataframe
        conditional_means_df = construct_results(
            self.paths['random_variables'], results_function=conditional_means)

        # Save the results to a .csv file
        conditional_means_df.to_csv(
            self.paths['results'] / 'conditional_means.csv', index=False)

        # Get daily, weekly, and yearly averages
        daily_results_df = construct_results(self.paths['random_variables'],
                                             results_function=intra_daily_averages)
        weekly_results_df = construct_results(self.paths['random_variables'],
                                              results_function=intra_weekly_averages)
        yearly_results_df = construct_results(self.paths['random_variables'],
                                              results_function=intra_year_averages)

        # Save to disk
        daily_results_df.to_csv(
            self.paths['results'] / 'daily_results.csv', index=False)
        weekly_results_df.to_csv(
            self.paths['results'] / 'weekly_results.csv', index=False)
        yearly_results_df.to_csv(
            self.paths['results'] / 'yearly_results.csv', index=False)

    def _compile_random_variables(self):
        # Initialize a list to store the random variables over the simulations
        random_variables_dict: list[dict] = []

        # Iterate over the simulations
        for run in self.list_simulations:
            # Check if the run was successful
            if not run.successful():
                logging.error("Run %s was not successful", run.name)
                continue

            # Create RunProcessor object
            run_processor = RunProcessor(run)

            # Check if the run was processed
            if not run_processor.processed_status():
                logging.error("Run %s was not processed", run.name)
                continue

            # Load random variables from the json results file
            random_variables: dict = run_processor.load_random_variables_df()

            random_variables_dict.append(random_variables)

        # Transform random variables into a pandas dataframe
        random_variables_df: pd.DataFrame = pd.DataFrame(random_variables_dict)

        return random_variables_df

    def _investment_results(self, lazy=True):
        rows: list = []
        for investment_problem in self.list_investment_problems:
            # get the current iteration object
            last_iteration: OptimizationPathEntry = get_last_successful_iteration(
                investment_problem.optimization_trajectory)

            convergence_reached: bool = last_iteration.check_convergence()
            iteration_number: int = last_iteration.iteration
            exo_vars: dict = {key: entry['value']
                              for key, entry in investment_problem.exogenous_variable.items()}

            if lazy:
                profits_dict: Optional[dict] = last_iteration.profits
            else:
                last_run = investment_problem.create_run(
                    last_iteration.current_investment)
                last_run_processor = RunProcessor(last_run)
                profits_dict = last_run_processor.get_profits()

            if profits_dict is None:
                logger.error("Unexpected: profits_dict is None for %s",
                             investment_problem.name)
                continue

            profits_with_suffix = {key + '_profit': value for key, value in
                                   profits_dict.items()}

            row: dict = {
                **exo_vars,
                **last_iteration.current_investment,
                'convergence_reached': convergence_reached,
                'iteration': iteration_number,
                **profits_with_suffix
            }

            # Append the new row to results_df
            rows.append(row)

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


def conditional_means(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    # Create cutoffs dictionary
    cutoffs = {
        'water_level_salto': {
            '25': run_df['water_level_salto'].quantile(0.25),
            '10': run_df['water_level_salto'].quantile(0.10)
        },
        'production_wind': {
            '25': run_df['production_wind'].quantile(0.25),
            '10': run_df['production_wind'].quantile(0.10)
        },
        'lost_load': {
            '95': run_df['lost_load'].quantile(0.95),
            '99': run_df['lost_load'].quantile(0.99)
        },
        'profits_thermal': {
            '75': run_df['profits_thermal'].quantile(0.75),
            '95': run_df['profits_thermal'].quantile(0.95)
        }
    }

    # Add cutoffs as columns to the dataframe
    for var, percentiles in cutoffs.items():
        for perc, value in percentiles.items():
            run_df[f'{var}_cutoff_{perc}'] = value

    queries_dict = {
        'unconditional': 'index==index',
        'water_level_34': f'water_level_salto < 34',
        'water_level_33': f'water_level_salto < 33',
        'water_level_32': f'water_level_salto < 32',
        'water_level_31': f'water_level_salto < 31',
        'drought_25': f'water_level_salto < water_level_salto_cutoff_25',
        'drought_10': f'water_level_salto < water_level_salto_cutoff_10',
        'low_wind_25': f'production_wind < production_wind_cutoff_25',
        'low_wind_10': f'production_wind < production_wind_cutoff_10',
        'drought_low_wind_25': f'water_level_salto < water_level_salto_cutoff_25 and production_wind < production_wind_cutoff_25',
        'drought_low_wind_10': f'water_level_salto < water_level_salto_cutoff_10 and production_wind < production_wind_cutoff_10',
        'blackout_95': f'lost_load > lost_load_cutoff_95',
        'blackout_99': f'lost_load > lost_load_cutoff_99',
        'negative_lost_load': f'lost_load < 0.001',
        'blackout_positive': f'lost_load > 0.001',
        'profits_thermal_75': f'profits_thermal > profits_thermal_cutoff_75',
        'profits_thermal_95': f'profits_thermal > profits_thermal_cutoff_95',
    }

    for query_name, query in queries_dict.items():
        query_frequency = run_df.query(
            query).shape[0] / run_df.shape[0]
        results_dict[f'{query_name}_frequency'] = query_frequency
        for variable in variables:
            results_dict[f'{query_name}_{variable}'] = run_df.query(query)[
                variable].mean()
    return results_dict


def intra_daily_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each hour of the day
    for hour in range(24):
        for variable in variables:
            run_df['hour'] = run_df['datetime'].dt.hour
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour']
                                                             == hour][variable].mean()

    return results_dict


def intra_weekly_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])
    # Take the mean of the variables for each hour of the week
    for hour in range(168):
        for variable in variables:
            run_df['hour_of_the_week'] = run_df['datetime'].dt.hour + \
                run_df['datetime'].dt.dayofweek * 24
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour_of_the_week']
                                                             == hour][variable].mean()

    return results_dict


def intra_year_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each day of the year
    for day in range(365):
        for variable in variables:
            run_df['day_of_the_year'] = run_df['datetime'].dt.dayofyear
            results_dict[f'{variable}_day_{day}'] = run_df[run_df['day_of_the_year']
                                                           == day][variable].mean()

    return results_dict
