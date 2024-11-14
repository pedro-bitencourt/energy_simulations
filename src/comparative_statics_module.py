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
from rich.logging import RichHandler
import numpy as np
import pandas as pd

from src.investment_module import InvestmentProblem
from src.run_module import Run
from src.run_processor_module import RunProcessor
from src.optimization_module import OptimizationPathEntry, get_last_successful_iteration

from src.constants import BASE_PATH

# Configure the root logger with rich handler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=True,
            markup=True
        )
    ]
)

# Get logger for current module
logger = logging.getLogger(__name__)


class ComparativeStatics:
    """
    Represents a comparative statics object. 
    Arguments:
        - name: name of the comparative statics exercise.
        - variables: dictionary containing the exogenous and endogenous variables.
        - variables_grid: dictionary containing the grids for the exogenous variables.
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
        - exogenous_variables: dictionary containing the exogenous variables.
        - endogenous_variables: dictionary containing the endogenous variables.
        - variables_grid: dictionary containing the grids for the exogenous variables.
        - paths: dictionary containing the paths for the input, output, and result folders.
        - list_simulations: list containing the Run objects.
        - list_investment_problems: list containing the InvestmentProblem objects.
    """

    def __init__(self,
                 name: str,
                 variables: Dict[str, Dict],
                 variables_grid: Dict[str, np.ndarray],
                 general_parameters: Dict):
        """
        Initialize the ComparativeStatics object.
        """
        self.name: str = name
        self.general_parameters: dict = general_parameters

        # Variables can be exogenous and endogenous
        self.exogenous_variables: dict[str,
                                       dict] = variables.get('exogenous', {})
        self.endogenous_variables: dict[str,
                                        dict] = variables.get('endogenous', {})
        self.variables_grid: dict[str, np.ndarray] = variables_grid

        # Initialize relevant paths
        self.paths: dict = self._initialize_paths()

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
            self.list_simulations = self._create_runs_from_investment_problems()
        else:
            # Create runs directly
            self.list_simulations = self._initialize_runs()

    def _validate_input(self):
        # Check if all variables have a grid
        for variable in self.exogenous_variables:
            if variable not in self.variables_grid:
                logging.error(f"Variable {variable} does not have a grid.")
                raise ValueError(f"Variable {variable} does not have a grid.")
        # Check if the grids have the same length
        if len(set([len(grid) for grid in self.variables_grid.values()])) != 1:
            logging.error("The grids have different lengths.")
            raise ValueError("The grids have different lengths.")
        # Check if general parameters contain the expected keys
        expected_keys = ['xml_basefile', 'daily', 'name_subfolder', 'email']
        if not all(key in self.general_parameters for key in expected_keys):
            logging.error(
                "General parameters do not contain the expected keys.")
            raise ValueError(
                "General parameters do not contain the expected keys.")

    def _create_runs_from_investment_problems(self):
        """
        Recover the last iteration of each investment problem and create a Run object from it.
        """
        list_simulations = []
        for investment_problem in self.list_investment_problems:
            # Load the last iteration as the Run
            last_iteration = get_last_successful_iteration(
                investment_problem.optimization_trajectory)
            if last_iteration is None:
                logging.critical(
                    f"No successful iteration found for investment problem {investment_problem.name}")
                continue
            # Create a Run object from the last iteration
            equilibrium_run = investment_problem.create_run(
                last_iteration.current_investment)

            # Clear the folder for the other runs
            logger.info("Deleting run folders for investment problem %s with equilibrium run %s",
                        investment_problem.name, equilibrium_run.name)
            investment_problem._clear_runs_folders(equilibrium_run.name)
            list_simulations.append(equilibrium_run)
        return list_simulations

    def _grid_length(self):
        return len(next(iter(self.variables_grid.values())))

    def _initialize_paths(self):
        paths = {}
        paths['main'] = Path(f"{BASE_PATH}/comparative_statics/{self.name}")
        paths['results'] = Path(f"{BASE_PATH}/results/{self.name}")
        return paths

    def _initialize_runs(self):
        # iterate over the grid
        list_simulations: list[Run] = []
        for i in range(self._grid_length()):
            variables = {
                key: {"pattern": self.exogenous_variables[key]["pattern"],
                      "value": self.variables_grid[key][i]}
                for key in self.exogenous_variables
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
            exogenous_variables_temp: dict = {
                variable: {
                    'value': self.variables_grid[variable][idx],
                    **var_dict
                }
                for variable, var_dict in self.exogenous_variables.items()
            }

            # initialize the InvestmentProblem object
            investment_problem = InvestmentProblem(self.paths['main'],
                                                   exogenous_variables_temp,
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
            # Process investment problems
            results_df = self._investment_results()

            # Save results of the optimization algorithm to CSV
            output_csv_path: Path = self.paths['results'] / \
                'investment_results.csv'
            results_df.to_csv(output_csv_path, index=False)

            # Create runs from investment problems
            self.list_simulations = self._create_runs_from_investment_problems()

        # Process runs
        self._process_runs()

    def _process_runs(self):
        # if experiment has more than 10 runs, submit jobs to cluster
        if len(self.list_simulations) > 10:
            process_locally: bool = False
        else:
            process_locally: bool = True

        for run in self.list_simulations:
            try:
                run_processor = RunProcessor(run)
                run_processor.process(process_locally)
                logging.info("Processed run %s", run.name)
            except ValueError:
                logging.critical("Run %s not successful", run.name)

        # gather the results
        results_df: pd.DataFrame = self._results()

        # save the results to a .csv file
        results_df.to_csv(
            self.paths['results'] / 'results_table.csv', index=False)

        # Log the results
        logging.info("Results for the comparative statics exercise %s: %s",
                     self.name, results_df)

    def _submit_processor_jobs(self):
        job_ids: list = []
        for run in self.list_simulations:
            run_processor = RunProcessor(run)
            if run_processor is None:
                logging.error("Run %s could not be processed.", run.name)
                continue
            if run_processor.processed_status():
                logging.info("Run %s already processed", run.name)
                continue

            job_id = run_processor.submit_processor_job()

            job_ids.append(job_id)
            logging.info("Submitted processor job with id %s for run %s",
                         job_id, run.name)
        return job_ids

    def _results(self):
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
                              for key, entry in investment_problem.exogenous_variables.items()}

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

            # Print the optimization trajectory
            investment_problem.print_optimization_trajectory()

        results_df: pd.DataFrame = pd.DataFrame(rows)

        return results_df

    ############################
    # NOT IN USE

    def _get_parameters(self):
        """
        Extract name of the run, the initial year and the final year
        from the .xml file
        """
        xml_basefile_filepath = self.general_parameters['xml_basefile']
        with open(xml_basefile_filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            name = content.split("<nombre>")[1].split("</nombre>")[0].strip()
            initial_year = int(content.split(
                "<inicioCorrida>")[1].split(" ")[2])
            final_year = int(content.split("<finCorrida>")[1].split(" ")[2])
        return name, initial_year, final_year
