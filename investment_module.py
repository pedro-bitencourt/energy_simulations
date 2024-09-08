"""
File name: investment_module.py
Description: this module creates a class that uses the Experiment module to compute, for a grid
of parameters, the optimal level of investment_problem in renewables (solar and wind) and the outcomes 
such as the total cost of the system. 
For now, the criterion for the optimal level of investment_problem is equating the average revenue per
MW over 20 years to the levelized cost of energy.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
import time
import logging
import subprocess
import json
import pandas as pd
import numpy as np
from run_module import Run
from run_processor_module import RunProcessor
import auxiliary
from constants import BASE_PATH

REQUESTED_TIME: str = '3:30:00'


logger = logging.getLogger(__name__)


def main():
    # initialize base parameters
    name: str = 'inv_zero_mc_thermal'
    xml_basefile: str = f'{BASE_PATH}/code/xml/inv_zero_mc_thermal.xml'

    run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                'thermal': {'position': 1, 'multiplier': 1},
                                'wind': {'position': 2, 'multiplier': 1},
                                'solar': {'position': 3, 'multiplier': 1}}

    general_parameters: dict = {'daily': False,
                                'name_subfolder': 'PRUEBA',
                                'xml_basefile': xml_basefile,
                                'name_function': run_name_function_params}

    # create the grid of exogenous variables
    current_hydro_capacity: int = 2215
    current_thermal_capacity_per_module: int = 45
    discrete_grid: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1]

    exogenous_variables: dict[str, dict] = {
        'hydro_factor': {'pattern': 'HYDRO_FACTOR'},
        'thermal': {'pattern': 'THERMAL_CAPACITY'}
    }
    endogenous_variables: dict[str, dict] = {
        'wind': {'pattern': 'WIND_CAPACITY'},
        'solar': {'pattern': 'SOLAR_CAPACITY'}
    }

    grid_hydro_factor: np.array = np.array(discrete_grid)
    grid_thermal_capacity: np.array = (
        current_thermal_capacity_per_module + (1-grid_hydro_factor)*current_hydro_capacity/6)
    # concatenate the two arrays
    exogenous_variables_grid: dict[str, np.array] = {'hydro_factor': grid_hydro_factor,
                                                     'thermal': grid_thermal_capacity}

    # create the experiment
    experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                      endogenous_variables, general_parameters)

    experiment.submit_jobs()
    # experiment.recover_results()


class InvestmentExperiment:
    '''
    The InvestmentExperiment class performs comparative statics with an endogenous capacity for some 
    energy sources.  
    '''

    def __init__(self, name: str,
                 exogenous_variables: dict[str, dict],
                 exogenous_variables_grid: np.array,
                 endogenous_variables: dict[str, dict],
                 general_parameters: dict):
        self.name: str = name
        self.output_folder: Path = Path(f'{BASE_PATH}/output/{name}')
        self.json_path: Path = Path(f'{BASE_PATH}/output/{name}.json')

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.list_investment_problems: list[InvestmentProblem] = self._create_investment_problems(
            exogenous_variables, exogenous_variables_grid, endogenous_variables, general_parameters)
        logging.debug(f"{self.list_investment_problems=}")

    def _create_investment_problems(self, exogenous_variables, exogenous_variables_grid,
                                    endogenous_variables, general_parameters):

        grid_lengths: list[int] = [
            len(grid) for grid in exogenous_variables_grid.values()]

        grid_length: int = max(grid_lengths)

        if grid_length > min(grid_lengths):
            print("ERROR: grid_length is not constant across variables.")

        problems = []
        for idx in range(grid_length):
            exogenous_variables_temp: dict = {}
            for variable, var_dict in exogenous_variables.items():
                exogenous_variables_temp[variable] = {
                    'pattern': var_dict['pattern'],
                    'value': exogenous_variables_grid[variable][idx]
                }

            logging.info('Initializing an InvestmentProblem object with exogenous variables %s',
                         exogenous_variables_temp)
            # initialize the InvestmentProblem object
            investment_problem = InvestmentProblem(self.name,
                                                   exogenous_variables_temp,
                                                   endogenous_variables,
                                                   general_parameters)

            logger.info(
                f'Created investment_problem object for {investment_problem.name}.')
            problems.append(investment_problem)
            del exogenous_variables_temp

        return problems

    def submit_jobs(self):
        logger.debug(f"{self.list_investment_problems=}")
        for inv_prob in self.list_investment_problems:
            inv_prob.run_on_quest()
            logger.info(f'Submitted job for {inv_prob.name}')

    def recover_results(self):
        rows: list = []
        for investment_problem in self.list_investment_problems:
            # get the current iteration object
            last_iteration: OptimizationTrajectoryEntry = investment_problem.optimization_trajectory[-1]
            convergence_reached: bool = check_convergence(last_iteration)
            iteration_number: int = last_iteration.iteration
            exo_vars: dict = {key: entry['value']
                              for key, entry in investment_problem.exogenous_variables.items()}

            profit: dict = last_iteration.profits['current']
            row: dict = {
                **exo_vars,
                **last_iteration.current_investment,
                'convergence_reached': convergence_reached,
                'iteration': iteration_number,
                'profit_wind': profit['wind'],
                'profit_solar': profit['solar']
            }

            # Append the new row to results_df
            rows.append(row)

        results_df: pd.DataFrame = pd.DataFrame(rows)

        print(f'{results_df=}')


@dataclass
class OptimizationTrajectoryEntry:
    iteration: int
    current_investment: dict
    successful: bool
    # convergence: bool
    profits: dict = None
    profits_derivatives: dict = None

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'current_investment': self.current_investment,
            'successful': self.successful,
            'profits': self.profits,
            'profits_derivatives': self.profits_derivatives
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class InvestmentProblem:
    def __init__(self,
                 name_experiment: str,
                 exogenous_variables: dict,
                 endogenous_variables: dict,
                 general_parameters: dict):
        self.name_experiment: str = name_experiment
        self.exogenous_variables: dict = exogenous_variables
        self.endogenous_variables: dict = endogenous_variables
        self.general_parameters: dict = general_parameters

        self.name = self.get_name()
        logging.info("Initializing investment problem %s", self.name)

        # initalize relevant paths
        self.paths: dict[str, Path] = self.initialize_paths()

        # initialize the optimization trajectory
        self.optimization_trajectory: list[OptimizationTrajectoryEntry] = self.initialize_optimization_trajectory(
        )
        logging.info("Successfully loaded optimization trajectory %s from filepath %s",
                     self.optimization_trajectory,
                     self.paths['optimization_trajectory'])

    def __repr__(self):
        return f"InvestmentProblem: exogenous variables {self.exogenous_variables}"

    def get_name(self):
        # hard coding now
        hydro_key: int = int(
            10*self.exogenous_variables['hydro_factor']['value'])
        thermal_key: int = int(self.exogenous_variables['thermal']['value'])
        name: str = f"{self.name_experiment}_{hydro_key}_{thermal_key}"
        return name

    def initialize_optimization_trajectory(self):
        # initialize the optimization trajectory
        optimization_trajectory: list[OptimizationTrajectoryEntry] = []

        if self.paths['optimization_trajectory'].exists():
            with open(self.paths['optimization_trajectory'], 'r') as file:
                data = json.load(file)
                optimization_trajectory = [
                    OptimizationTrajectoryEntry.from_dict(entry) for entry in data]
            logging.info("Successfully loaded optimization trajectory.")
            logging.info("Optimization trajectory: %s",
                         optimization_trajectory)
        else:
            logging.info(
                f"Optimization trajectory not found at {self.paths['optimization_trajectory']}.")
            iteration_0 = OptimizationTrajectoryEntry(
                iteration=0,
                current_investment={'wind': 2000, 'solar': 250},
                successful=False,
            )
            optimization_trajectory.append(iteration_0)
        return optimization_trajectory

    def save_optimization_trajectory(self):
        with open(self.paths['optimization_trajectory'], 'w') as file:
            json.dump([entry.to_dict() for entry in self.optimization_trajectory],
                      file, indent=4, sort_keys=True)
        logging.info(
            f'Saved optimization trajectory to {self.paths["optimization_trajectory"]}')

    def initialize_paths(self):
        paths: dict[str, Path] = {}
        paths['experiment_folder'] = Path(
            f'{BASE_PATH}/output/{self.name_experiment}')
        paths['output'] = paths['experiment_folder'] / Path(self.name)

        paths['input'] = Path(f'{BASE_PATH}/temp/investment/{self.name}')
        paths['optimization_trajectory'] = paths['experiment_folder'] /\
            Path(f'{self.name}_trajectory.json')

        paths['output'].mkdir(parents=True, exist_ok=True)
        paths['input'].mkdir(parents=True, exist_ok=True)

        return paths

    def solve_investment(self):
        # initialize the current investment as the last element of the optimization trajectory
        current_iteration: OptimizationTrajectoryEntry = self.optimization_trajectory[-1]
        logger.info('Initializing the solver at iteration %s',
                    current_iteration)
        # set the maximum number of iterations and tolerance
        max_iter: int = 10

        while current_iteration.iteration < max_iter:
            logger.info('Current iteration: %s', current_iteration)
            # force output
            sys.stdout.flush()
            sys.stderr.flush()

            # compute the profits and derivatives
            current_iteration = self.compute_iteration_profits(
                current_iteration)

            # force output
            sys.stdout.flush()
            sys.stderr.flush()

            # update the optimization trajectory
            self.optimization_trajectory[-1] = current_iteration

            # save the optimization trajectory
            self.save_optimization_trajectory()

            # check for convergence
            if check_convergence(current_iteration):
                logger.info(
                    'Convergence reached. Current iteration %s', current_iteration)
                # current_iteration.convergence = True
                # self.optimization_trajectory[-1] = current_iteration
                # save results
                self.save_optimization_trajectory()

            # compute the new investment
            new_iteration: OptimizationTrajectoryEntry = newton_iteration(
                current_iteration)

            # update the optimization trajectory
            self.optimization_trajectory.append(new_iteration)
            self.save_optimization_trajectory()

            # update the current iteration
            current_iteration: OptimizationTrajectoryEntry = new_iteration

        logger.info(
            f'Maximum number of iterations reached. Optimization trajectory saved.')
        # save results
        self.save_optimization_trajectory()

    def compute_iteration_profits(self, current_iteration: OptimizationTrajectoryEntry) -> OptimizationTrajectoryEntry:
        # check if the current iteration was successful
        if current_iteration.successful:
            current_investment = current_iteration.current_investment
            profits = current_iteration.profits
            profits_derivatives = current_iteration.profits_derivatives
        else:
            # compute the profits and derivatives
            current_investment = current_iteration.current_investment
            profits, profits_derivatives = self.profits_and_derivatives(
                current_investment)

        # Log the results
        logger.info(
            f'investment in wind: {current_investment["wind"]} MW')
        logger.info(
            f'investment in solar: {current_investment["solar"]} MW')
        logger.info(f'Profits: {profits}')
        logger.info(f'Profits derivatives: {profits_derivatives}')

        # Update the entry
        return OptimizationTrajectoryEntry(
            iteration=current_iteration.iteration,
            current_investment=current_investment,
            successful=True,
            profits=profits,
            profits_derivatives=profits_derivatives
        )

    def profits_and_derivatives(self, current_investment: dict):
        # use 50 MW as the delta for the derivatives
        delta = 50
        renewables = {'w': 'wind', 's': 'solar'}

        job_ids = []

        # submit run at the current investment
        run_current: Run = self.create_run(current_investment)
        job_ids.append(run_current.submit_run(REQUESTED_TIME))
        # submit runs at the current investment_problem plus delta
        run_plus: dict[str, Run] = {}
        for key, renewable in renewables.items():
            new_investment = current_investment.copy()
            new_investment[renewable] += delta
            run_plus[renewable]: Run = self.create_run(new_investment)
            job_ids.append(run_plus[renewable].submit_run(REQUESTED_TIME))

        # wait for jobs to be completed to get profits
        wait_for_jobs(job_ids)

        # force output
        sys.stdout.flush()
        sys.stderr.flush()

        # get the profits from the output files
        profits: dict = {}
        run_current_processor: RunProcessor = RunProcessor(run_current)
        profits['current']: dict = run_current_processor.get_profits()
        for renewable in renewables.values():
            run_plus_processor: RunProcessor = RunProcessor(
                run_plus[renewable])
            profits[renewable]: dict = run_plus_processor.get_profits()

        # compute the derivatives
        derivatives = derivatives_from_profits(profits, delta)

        return profits, derivatives

    def create_run(self, current_investment: dict):
        """
        Creates a Run object from a level of current investment_problem in wind 
        and solar
        """
        # create a values array with the same order as the variables
        endogenous_variables_temp: dict = self.endogenous_variables.copy()
        for key, entry in endogenous_variables_temp.items():
            endogenous_variables_temp[key]['value'] = current_investment[key]

        variables: dict = self.exogenous_variables.copy()
        variables.update(endogenous_variables_temp)

        # create the Run object
        run: Run = Run(self.paths['input'],
                       self.general_parameters,
                       variables,
                       experiment_folder=self.paths['output'])
        return run

    def run_on_quest(self):
        logging.info("Preparing to run %s on quest",
                     self.name)
        logging.info(
            f"Exogenous variables: {self.exogenous_variables}")
        investment_data = {
            "name_experiment": self.name_experiment,
            "exogenous_variables": self.exogenous_variables,
            "endogenous_variables": self.endogenous_variables,
            "general_parameters": self.general_parameters
        }

        logging.info(f"Investment data: {investment_data}")

        output_folder = self.paths['output']
        input_folder = self.paths['input']
        script_path = f"{output_folder}/{self.name}_script.sh"
        data_path = f"{input_folder}/{self.name}_data.json"

        # Write investment data to a separate JSON file
        with open(data_path, 'w') as f:
            json.dump(investment_data, f)

        self.create_bash_script(data_path, script_path)
        job_id = auxiliary.submit_slurm_job(script_path)

        del investment_data

        return job_id

    def create_bash_script(self, data_path: str, script_path: str):
        with open(script_path, 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --job-name={self.name}
#SBATCH --output={self.paths['experiment_folder']}/{self.name}.out
#SBATCH --error={self.paths['experiment_folder']}/{self.name}.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0

python - <<END

print('Solving the investment_problem problem {self.name}')

import sys
import json
sys.path.append('/projects/p32342/code')
from investment_module import InvestmentProblem

with open('{data_path}', 'r') as f:
    investment_data = json.load(f)

investment_problem = InvestmentProblem(**investment_data)
print('Successfully loaded the investments problem data')
sys.stdout.flush()
sys.stderr.flush()
investment_problem.solve_investment()
print(f'InvestmentProblem calculation completed:')
print(f'{{investment_problem.optimization_trajectory=}}')
END
''')


def check_convergence(current_entry: OptimizationTrajectoryEntry) -> bool:
    profits = current_entry.profits
    if profits is None:
        return False

    threshold_wind: int = 10_000
    threshold_solar: int = 10_000

    curr_prof_wind = np.abs(profits['current']['wind'])
    curr_prof_solar = np.abs(profits['current']['solar'])
    return curr_prof_wind < threshold_wind and curr_prof_solar < threshold_solar


def newton_iteration(current_entry: OptimizationTrajectoryEntry) -> OptimizationTrajectoryEntry:
    profits = current_entry.profits
    profits_derivatives = current_entry.profits_derivatives
    current_investment = current_entry.current_investment

    # Transform the dictionaries into np.arrays
    profits_array = np.array(
        [profits['current']['wind'], profits['current']['solar']])
    profits_derivatives_array = np.array([
        [profits_derivatives['w']['w'], profits_derivatives['w']['s']],
        [profits_derivatives['s']['w'], profits_derivatives['s']['s']]
    ])
    current_investment_array = np.array(
        [current_investment['wind'], current_investment['solar']])

    # Compute the new investment using numpy operations
    try:
        investment_change = np.linalg.solve(
            profits_derivatives_array, profits_array)
        new_investment_array = current_investment_array - investment_change
    except np.linalg.LinAlgError:
        # Handle singular matrix error
        logger.warning(
            "Singular matrix encountered. Using pseudoinverse instead.")
        investment_change = np.linalg.pinv(
            profits_derivatives_array) @ profits_array
        new_investment_array = current_investment_array - investment_change

    # Round new_investment to nearest 10
    new_investment_array = np.round(new_investment_array, -1)

    # Transform the new investment into a dictionary
    new_investment_dict: dict[str, float] = {
        'wind': float(new_investment_array[0]),
        'solar': float(new_investment_array[1])
    }

    # Create and return a new OptimizationTrajectoryEntry
    return OptimizationTrajectoryEntry(
        iteration=current_entry.iteration + 1,
        current_investment=new_investment_dict,
        successful=False,  # This will be set to True after profits are computed
        profits=None,
        profits_derivatives=None
    )


def derivatives_from_profits(profits: dict, delta: int):
    # initalize dictionary of derivatives
    # convention: derivatives['w']['s'] = dprofit_wind/dsolar
    renewables = {'w': 'wind', 's': 'solar'}
    derivatives: dict = {'w': {'w': 0, 's': 0}, 's': {'w': 0, 's': 0}}
    for key1 in derivatives.keys():
        for key2 in derivatives[key1].keys():
            ren_1 = renewables[key1]
            ren_2 = renewables[key2]
            derivative_temp = ((profits[ren_1][ren_2] - profits['current'][ren_1])
                               / delta)
            derivatives[key1][key2] = derivative_temp
    return derivatives


def get_last_iteration(optimization_trajectory):
    return max(optimization_trajectory, key=lambda obj: obj.iteration)


def check_job_status(job_id: int):
    result = subprocess.run(
        ['squeue', '-j', str(job_id), '-h'], capture_output=True, text=True)
    return len(result.stdout.strip()) > 0


def wait_for_jobs(job_ids: list):
    job_ids = [job_id for job_id in job_ids if job_id is not True]
    while True:
        if all(not check_job_status(job_id) for job_id in job_ids):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
