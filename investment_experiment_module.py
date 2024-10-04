
import logging
import sys
from pathlib import Path
from typing import Optional
import time
import numpy as np
import pandas as pd
from investment_module import InvestmentProblem
from run_module import Run
from experiment_module import Experiment
from optimization_module import OptimizationPathEntry, get_last_successful_iteration

from constants import BASE_PATH


logger = logging.getLogger(__name__)


def main():
    mrs_list = [0.25, 0.5, 0.75]
    for mrs in mrs_list:
        # initialize base parameters
        mrs = 0.5
        name: str = f'zero_mc_thermal_mrs_{mrs}'
        xml_basefile: str = f'{BASE_PATH}/code/xml/zero_mc_thermal.xml'

        run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                    'thermal': {'position': 1, 'multiplier': 1},
                                    'wind': {'position': 2, 'multiplier': 1},
                                    'solar': {'position': 3, 'multiplier': 1}}

        general_parameters: dict = {'daily': True,
                                    'name_subfolder': 'CAD-2024-DIARIA',
                                    'xml_basefile': xml_basefile,
                                    'name_function': run_name_function_params}

        # create the grid of exogenous variables
        current_hydro_capacity: int = 2215
        current_thermal_capacity_per_module: int = 0

        # discrete_grid: list[float] = [1]
        discrete_grid: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        exogenous_variables: dict[str, dict] = {
            'hydro_factor': {'pattern': 'HYDRO_FACTOR'},
            'thermal': {'pattern': 'THERMAL_CAPACITY'}
        }
        endogenous_variables: dict[str, dict] = {
            'wind': {'pattern': 'WIND_CAPACITY'},
            'solar': {'pattern': 'SOLAR_CAPACITY'}
        }

        grid_hydro_factor: np.ndarray = np.array(discrete_grid)
        grid_thermal_capacity: np.ndarray = mrs*(
            current_thermal_capacity_per_module + (1-grid_hydro_factor)*current_hydro_capacity/6)
        # concatenate the two arrays
        exogenous_variables_grid: dict[str, np.ndarray] = {'hydro_factor': grid_hydro_factor,
                                                           'thermal': grid_thermal_capacity}

        # create the experiment
        experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                          endogenous_variables, general_parameters)

        # results_df = experiment.recover_results()
        # print("Results so far:")
        # print(f'{results_df=}')

        experiment.submit_jobs()
        del experiment

    # experiment.process_results()


class InvestmentExperiment:
    '''
    The InvestmentExperiment class performs comparative statics with an endogenous capacity for some
    energy sources.
    '''

    def __init__(self, name: str,
                 exogenous_variables: dict[str, dict],
                 exogenous_variables_grid: dict[str, np.ndarray],
                 endogenous_variables: dict[str, dict],
                 general_parameters: dict):
        self.name: str = name
        self.output_folder: Path = Path(f'{BASE_PATH}/output/{name}')
        self.json_path: Path = Path(f'{BASE_PATH}/output/{name}.json')
        self.general_parameters: dict = general_parameters

        self.exogenous_variables: dict[str, dict] = exogenous_variables
        self.endogenous_variables: dict[str, dict] = endogenous_variables

        # create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # create list of investment problems
        self.list_investment_problems: list[InvestmentProblem] = self._create_investment_problems(
            exogenous_variables, exogenous_variables_grid, endogenous_variables, general_parameters)

    def _create_investment_problems(self, exogenous_variables, exogenous_variables_grid,
                                    endogenous_variables, general_parameters):
        # get the grid length
        grid_lengths: list[int] = [
            len(grid) for grid in exogenous_variables_grid.values()]
        grid_length: int = grid_lengths[0]

        # check if the grid length is constant across variables
        equal_lengths: bool = all(
            length == grid_length for length in grid_lengths)
        if not equal_lengths:
            logger.critical(
                "CRITICAL: grid_length is not constant across variables. Aborting.")
            sys.exit(1)

        # create a list of InvestmentProblem objects
        problems = []
        for idx in range(grid_length):
            # get the exogenous variables for the current iteration
            exogenous_variables_temp: dict = {
                variable: {
                    'pattern': var_dict['pattern'],
                    'value': exogenous_variables_grid[variable][idx]
                }
                for variable, var_dict in exogenous_variables.items()
            }

            # initialize the InvestmentProblem object
            investment_problem = InvestmentProblem(self.name,
                                                   exogenous_variables_temp,
                                                   endogenous_variables,
                                                   general_parameters)

            logger.info(
                f'Created investment_problem object for {investment_problem.name}.')
            problems.append(investment_problem)

            # avoid memory leak
            del exogenous_variables_temp
        return problems

    def submit_jobs(self):
        for inv_prob in self.list_investment_problems:
            inv_prob.run_on_quest()
            logger.info(f'Submitted job for {inv_prob.name}')

    def process_results(self):
        # recover optimization results
        results_df: pd.DataFrame = self.recover_results()

        # save optimization results to csv
        output_csv_path: Path = self.output_folder / \
            f'{self.name}_opt_results.csv'
        results_df.to_csv(output_csv_path, index=False)

        # create experiment from investment experiment
        experiment: Experiment = self.experiment_from_investment_experiment()

        # process as experiment
        experiment.process_experiment()

        # visualize results
        # experiment.visualize_experiment()

    def visualize_results(self):
        experiment: Experiment = self.experiment_from_investment_experiment()
        experiment.visualize_experiment(grid_dimension=2)

    def experiment_from_investment_experiment(self):
        equilibrium_runs_array: list[Run] = []
        for investment_problem in self.list_investment_problems:
            equilibrium_run = investment_problem.equilibrium_run()
            if equilibrium_run is not None:
                equilibrium_runs_array.append(equilibrium_run)
        # create a Experiment instance
        variables: dict = {**self.exogenous_variables,
                           **self.endogenous_variables}
        experiment = Experiment(self.name,
                                variables,
                                self.general_parameters,
                                runs_array=equilibrium_runs_array)
        return experiment

    def recover_results(self):
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


if __name__ == '__main__':
    main()
