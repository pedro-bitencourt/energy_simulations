"""
Module: comparative_statics_module.py

This module contains the ComparativeStatics class, which abstracts both Experiment and InvestmentExperiment classes.
It is used to perform comparative statics experiments with the flexibility to handle both standard experiments and
investment experiments with endogenous capacities.

Public methods:
- submit: submits the jobs for the runs or investment problems.
- process: processes the results of the runs or investment problems.
- results: gathers the results of the runs or investment problems.

Required modules:
- investment_module: contains the InvestmentProblem class.
- run_module: contains the Run class.
- run_processor_module: contains the RunProcessor class.
- visualize_module: contains visualization functions.
- optimization_module: contains the OptimizationPathEntry class.
- auxiliary: contains auxiliary functions.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd

from src.investment_module import InvestmentProblem
from src.run_module import Run
from src.run_processor_module import RunProcessor
from src.optimization_module import OptimizationPathEntry, get_last_successful_iteration

from src.constants import BASE_PATH

# Get logger for current module
logger = logging.getLogger(__name__)


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
            o years_run: number of years to run the investment problems.
            o requested_time_run: requested time for each MOP run to the cluster.

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

        # Variables can be exogenous and endogenous
        self.exogenous_variable: dict[str,
                                       dict] = variables.get('exogenous', {})
        self.exogenous_variable_key: str = list(
            self.exogenous_variable.keys())[0]
        self.endogenous_variables: dict[str,
                                        dict] = variables.get('endogenous', {})

        self.exogenous_variable_grid: dict[str, np.ndarray] = {var_name: np.array(variable['grid'])
            for var_name, variable in self.exogenous_variable.items()}

        # Initialize relevant paths
        self.paths: dict = self._initialize_paths(base_path)

        # Create the folders if they do not exist
        for path in self.paths.values():
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

    def redo_equilibrium_runs(self):
        """
        Deletes and runs again the equilibrium runs.
        """
        for run in self.list_simulations:
            logger.info("Redoing equilibrium run %s", run.name)
            run.tear_down()
            run.submit()

    def get_dataframe(self, data_type: str, participant: str = '') -> pd.DataFrame:
        """
        Generic method to get dictionary of DataFrames for different data types.

        Args:
            data_type: Type of data to retrieve ('water_level', 'production', 'marginal_costs')
            participant: Participant name (optional, not needed for marginal_costs)

        Returns:
            DataFrame with the data for each run.
        """
        method_map = {
            'water_level': lambda proc, p: proc.water_level_participant(p),
            'production': lambda proc, p: proc.production_participant(p),
            'variable_costs': lambda proc, p: proc.variable_costs_participant(p),
            'marginal_cost': lambda proc, _: proc.marginal_cost_df()
        }

        if data_type not in method_map:
            raise ValueError(
                f"Invalid data type. Must be one of {list(method_map.keys())}")

        df_dict: dict[str, pd.DataFrame] = {}
        method = method_map[data_type]

        for run in self.list_simulations:
            try:
                run_processor = RunProcessor(run)
                df = method(run_processor, participant)
                df['run'] = run.name
                df_dict[run.name] = df
            except FileNotFoundError:
                logger.error(
                    f"{data_type.replace('_', ' ').title()} file not found for run {run.name}")
                continue
            except ValueError:
                logger.error(
                    f"Run {run.name} not successful, skipping it")
                continue

        # Merge the DataFrames into a single DataFrames
        df_merged = pd.concat(
            df_dict.values(), keys=df_dict.keys(), names=['run'])
        return df_merged

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
        logging.critical(
            f"Creating runs with check_convergence={check_convergence}")
        list_simulations = []
        for investment_problem in self.list_investment_problems:
            # Load the last iteration as the Run
            last_iteration = get_last_successful_iteration(
                investment_problem.optimization_trajectory)
            if last_iteration is None:
                logging.critical(
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

    def _grid_length(self):
        return len(next(iter(self.exogenous_variable_grid.values())))

    def _initialize_paths(self, base_path: str) -> dict:
        paths = {}
        paths['main'] = Path(f"{base_path}/comparative_statics/{self.name}")
        paths['results'] = Path(f"{base_path}/results/{self.name}")
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

            # initialize the InvestmentProblem object
            investment_problem = InvestmentProblem(self.paths['main'],
                                                   exogenous_variable_temp,
                                                   self.endogenous_variables,
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

    def process(self):
        """
        Create runs from investment problems, if there are endogenous variables, and process the results.
        """
        # If there are endogenous_variables, first process investment problems and create
        # a list of Run objects
        if self.endogenous_variables:
            # Create runs from investment problems
            self.list_simulations = self.create_runs_from_investment_problems()

        # Process the runs
        self._process_runs()

        # Compile the dataframes
        self._compile_dataframes()

        # Construct the new results dataframe
        new_results_df = self.construct_new_results()

        # Save the results to a .csv file
        new_results_df.to_csv(
            self.paths['results'] / 'new_results_table.csv', index=False)

        # Construct the results table
        results_df = self._construct_results_table()

        # Log the results
        logging.info("Results for the comparative statics exercise %s: %s",
                     self.name, results_df)

        # Save the results to a .csv file
        results_df.to_csv(
            self.paths['results'] / 'results_table.csv', index=False)

        # Compile the random_variables_df
        random_variables_df = self._compile_random_variables()
        # Save
        random_variables_df.to_csv(
            self.paths['results'] / 'random_variables.csv', index=False)

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

    def construct_new_results(self):
        first_flag = True
        for inv_prob in self.list_investment_problems:
            equilibrium_run, converged = inv_prob.equilibrium_run() 
            processor = RunProcessor(equilibrium_run)

            results_dict = processor.construct_results_dict()
            if first_flag:
                new_results_df = pd.DataFrame([results_dict])
                first_flag = False
            else:
                new_results_df = new_results_df.append(results_dict, ignore_index=True)

        return new_results_df



    def _compile_dataframes(self):
        # Extract comparative statics dataframe
        dataframes_to_extract: dict = {'water_level': ['salto'],
                                       'production': ['wind', 'solar', 'thermal'],
                                       'variable_costs': ['thermal'],
                                       }
        for data_type, participants in dataframes_to_extract.items():
            for participant in participants:
                df = self.get_dataframe(data_type, participant)
                df.to_csv(self.paths['results'] /
                          f'{data_type}_{participant}.csv')

        df = self.get_dataframe('marginal_cost')
        df.to_csv(self.paths['results'] / 'marginal_cost.csv')

    def _process_runs(self):
        # Process the runs
        for run in self.list_simulations:
            try:
                run_processor = RunProcessor(run)
                run_processor.process()
                logging.info("Processed run %s", run.name)
            except ValueError:
                logging.critical("Run %s not successful", run.name)


    def _construct_results_table(self):
        # initialize a list to store the results over the simulations
        results_dict: list[dict] = []

        # iterate over the simulations
        for run in self.list_simulations:
            # check if the run was successful
            if not run.successful():
                logging.error("Run %s was not successful", run.name)
                continue

            # create RunProcessor object
            run_processor = RunProcessor(run)

            # check if the run was processed
            if not run_processor.processed_status():
                logging.error("Run %s was not processed", run.name)
                continue

            # load results from the json results file
            results: dict = run_processor.load_results()

            results_dict.append(results)

        # transform results into a pandas dataframe
        results_df: pd.DataFrame = pd.DataFrame(results_dict)

        # If there are endogenous variables, compile results for the solver
        if self.endogenous_variables:
            # Process investment problems
            investment_results_df = self._investment_results()
            # Merge the solver's results into results_df
            results_df = pd.merge(results_df, investment_results_df,
                                  on=[self.exogenous_variable_key])

        return results_df

    def _investment_results(self):
        rows: list = []
        for investment_problem in self.list_investment_problems:
            # get the current iteration object
            last_iteration: OptimizationPathEntry = get_last_successful_iteration(
                investment_problem.optimization_trajectory)

            convergence_reached: bool = last_iteration.check_convergence()
            iteration_number: int = last_iteration.iteration
            exo_vars: dict = {key: entry['value']
                              for key, entry in investment_problem.exogenous_variable.items()}

            profits_dict: Optional[dict] = last_iteration.profits

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
