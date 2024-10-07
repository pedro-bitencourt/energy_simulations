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
import logging
import json
import numpy as np

from run_module import Run
from run_processor_module import RunProcessor
from optimization_module import (OptimizationPathEntry,
                                 derivatives_from_profits,
                                 print_optimization_trajectory_function,
                                 get_last_successful_iteration)
from auxiliary import submit_slurm_job, wait_for_jobs
from constants import BASE_PATH, REQUESTED_TIME_RUN

MAX_ITER: int = 15
UNSUCCESSFUL_RUN: int = 2

INITIAL_GUESS_WIND: int = 1800
INITIAL_GUESS_SOLAR: int = 500

logger = logging.getLogger(__name__)


def main():
    # initialize base parameters
    name: str = 'inv_zero_mc_thermal'
    xml_basefile: str = f'{BASE_PATH}/code/xml/inv_zero_mc_thermal.xml'

    run_name_function_params: dict = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                      'thermal': {'position': 1, 'multiplier': 1},
                                      'wind': {'position': 2, 'multiplier': 1},
                                      'solar': {'position': 3, 'multiplier': 1}}

    general_parameters: dict = {'daily': False,
                                'name_subfolder': 'PRUEBA',
                                'xml_basefile': xml_basefile,
                                'name_function': run_name_function_params}

    exogenous_variables: dict[str, dict] = {
        'hydro_factor': {'pattern': 'HYDRO_FACTOR', 'value': 1},
        'thermal': {'pattern': 'THERMAL_CAPACITY', 'value': 45}
    }
    endogenous_variables: dict[str, dict] = {
        'wind': {'pattern': 'WIND_CAPACITY'},
        'solar': {'pattern': 'SOLAR_CAPACITY'}
    }

    # create the investment problem
    investment_problem = InvestmentProblem(name,
                                           exogenous_variables,
                                           endogenous_variables,
                                           general_parameters)

    investment_problem.print_optimization_trajectory()
    investment_problem.solve_investment()


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
        self.optimization_trajectory: list = self.initialize_optimization_trajectory()

    def __repr__(self):
        return f"InvestmentProblem: exogenous variables {self.exogenous_variables}"

    def initialize_paths(self):
        paths: dict[str, Path] = {}
        paths['experiment_folder'] = Path(
            f'{BASE_PATH}/output/{self.name_experiment}')
        paths['output'] = paths['experiment_folder'] / Path(self.name)
        paths['bash_script'] = paths['output'] / Path(f'{self.name}.sh')
        paths['input'] = Path(f'{BASE_PATH}/temp/investment/{self.name}')
        paths['optimization_trajectory'] = paths['experiment_folder'] /\
            Path(f'{self.name}_trajectory.json')

        # create the directories
        paths['output'].mkdir(parents=True, exist_ok=True)
        paths['input'].mkdir(parents=True, exist_ok=True)

        return paths

    def get_name(self):
        # hard coding now
        hydro_key: int = int(
            10*self.exogenous_variables['hydro_factor']['value'])
        thermal_key: int = int(self.exogenous_variables['thermal']['value'])
        name: str = f"{self.name_experiment}_{hydro_key}_{thermal_key}"
        return name

    def initialize_optimization_trajectory(self):
        # check if the optimization trajectory file exists
        if self.paths['optimization_trajectory'].exists():
            # if yes, load it
            with open(self.paths['optimization_trajectory'], 'r') as file:
                data = json.load(file)
                optimization_trajectory = [OptimizationPathEntry.from_dict(entry)
                                           for entry in data]

            logging.info("Successfully loaded optimization trajectory from %s.",
                         self.paths['optimization_trajectory'])
            logging.debug("Optimization trajectory: %s",
                          optimization_trajectory)
        else:
            # if not, initialize it with the first iteration
            optimization_trajectory: list[OptimizationPathEntry] = []
            logging.info("Optimization trajectory not found at %s. Initializing a new one.",
                         self.paths['optimization_trajectory'])

            # initialize the first iteration
            iteration_0 = OptimizationPathEntry(
                iteration=0,
                current_investment={
                    'wind': INITIAL_GUESS_WIND, 'solar': INITIAL_GUESS_SOLAR},
                successful=False,
            )
            optimization_trajectory.append(iteration_0)
        return optimization_trajectory

    def save_optimization_trajectory(self):
        with open(self.paths['optimization_trajectory'], 'w') as file:
            json.dump([entry.to_dict() for entry in self.optimization_trajectory],
                      file, indent=4, sort_keys=True)
        logging.info('Saved optimization trajectory to %s',
                     self.paths["optimization_trajectory"])

    def solve_investment(self):
        # initialize the current investment as the last element of the optimization trajectory
        current_iteration: OptimizationPathEntry = self.optimization_trajectory[-1]
        logger.info('Initializing the solver at iteration %s',
                    current_iteration)

        # start the optimization loop
        while current_iteration.iteration < MAX_ITER:
            logger.info('Current iteration: %s', current_iteration)

            # force output
            sys.stdout.flush()
            sys.stderr.flush()

            # compute the profits and derivatives
            current_iteration = self.update_current_iteration(
                current_iteration)

            # update the optimization trajectory
            self.optimization_trajectory[-1] = current_iteration

            # save the optimization trajectory
            self.save_optimization_trajectory()

            # check for convergence
            if current_iteration.check_convergence():
                logger.info(
                    'Convergence reached. Current iteration %s', current_iteration)
                break

            # compute the new investment
            new_iteration: OptimizationPathEntry = current_iteration.next_iteration()

            # update the current iteration loop variable
            current_iteration: OptimizationPathEntry = new_iteration

            # append the new iteration to the optimization trajectory
            self.optimization_trajectory.append(current_iteration)

        logger.info(
            'Maximum number of iterations reached. Optimization trajectory saved.')

        # save results
        self.save_optimization_trajectory()

    def update_current_iteration(self,
                                 current_iteration: OptimizationPathEntry) -> OptimizationPathEntry:
        '''
        Updates the current iteration with the profits and derivatives
        '''
        current_investment = current_iteration.current_investment

        # check if the current iteration was successful
        if current_iteration.profits is None or current_iteration.profits_derivatives is None:
            logger.info('Preparing to compute profits for iteration with %s',
                        current_investment)
            # compute the profits and derivatives
            profits, profits_derivatives = self.profits_and_derivatives(
                current_investment)

        # if the profits are already computed, use them
        else:
            profits = current_iteration.profits
            profits_derivatives = current_iteration.profits_derivatives

        # output a new OptimizationPathEntry
        return OptimizationPathEntry(
            iteration=current_iteration.iteration,
            current_investment=current_investment,
            successful=True,
            profits=profits,
            profits_derivatives=profits_derivatives
        )

    def profits_and_derivatives(self, current_investment: dict) -> tuple[dict, dict]:
        '''
        Computes the profits and derivatives for a given level of investment_problem 
        in wind and solar
        '''
        delta = 50

        def create_and_submit_run(investment):
            run = self.create_run(investment)
            return run, run.submit_run(REQUESTED_TIME_RUN)

        # create dict of investments
        investments_dict: dict = {
            'current': current_investment,
            'wind': {
                'wind': current_investment['wind'] + delta,
                'solar': current_investment['solar']
            },
            'solar': {
                'wind': current_investment['wind'],
                'solar': current_investment['solar'] + delta
            }
        }

        # create and submit runs
        runs_and_jobs = {
            renewable: create_and_submit_run(investment)
            for renewable, investment in investments_dict.items()
        }

        # wait for all jobs to complete
        wait_for_jobs([job for _, job in runs_and_jobs.values()])

        # process results
        profits_perturb = {}
        for renewable, (run, _) in runs_and_jobs.items():
            run_processor = RunProcessor(run)
            # check if run was successful
            if run_processor:
                profits_perturb[renewable] = run_processor.get_profits()
            # if not, abort
            else:
                logging.critical(
                    "Run %s was not successful. Aborting.", run.run_name)
                sys.exit(UNSUCCESSFUL_RUN)

        # compute derivatives from the profits
        derivatives = derivatives_from_profits(profits_perturb, delta)
        return profits_perturb['current'], derivatives

    def create_run(self, current_investment: dict):
        """
        Creates a Run object from a level of current investment_problem in wind 
        and solar
        """
        # create a values array with the same order as the variables
        endogenous_variables_temp: dict = {
            key:
                {
                    'pattern': entry['pattern'],
                    'value': current_investment[key]
                }
            for key, entry in self.endogenous_variables.items()
        }

        # create the variables dictionary
        variables: dict = self.exogenous_variables.copy()
        variables.update(endogenous_variables_temp)

        # create the Run object
        run: Run = Run(self.paths['input'],
                       self.general_parameters,
                       variables,
                       experiment_folder=self.paths['output'])

        return run

    def print_optimization_trajectory(self):
        print_optimization_trajectory_function(self.optimization_trajectory)

    #########################################
    # Methods to process the results
    def process_results(self):
        # get the last successful iteration
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.optimization_trajectory)
        # check if the last iteration converged
        if last_iteration.check_convergence():
            # initialize a dictionary to store the results
            results_dict: dict = {}

            # get results on the iterative algorithm
            convergence_reached: bool = last_iteration.check_convergence()
            iteration_number: int = last_iteration.iteration
            exo_vars: dict = {key: entry['value']
                              for key, entry in self.exogenous_variables.items()}

            return results_dict
        else:
            return False

    def equilibrium_run(self):
        """
        Returns the equilibrium run if the last iteration converged
        """
        # get the last successful iteration
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.optimization_trajectory)
        # check if the last iteration converged
        if last_iteration.check_convergence():
            # create the run object
            run: Run = self.create_run(last_iteration.current_investment)
            return run
        return None

    #########################################
    # Methods to submit the job to Quest
    def run_on_quest(self):
        logging.info("Preparing to run %s on quest",
                     self.name)
        logging.debug("Exogenous variables: %s",
                      self.exogenous_variables)
        investment_data = {
            "name_experiment": self.name_experiment,
            "exogenous_variables": self.exogenous_variables,
            "endogenous_variables": self.endogenous_variables,
            "general_parameters": self.general_parameters
        }

        logging.debug("Investment data: %s", investment_data)

        script_path: Path = self.create_bash_script()
        job_id = submit_slurm_job(script_path)

        # avoid memory leak
        del investment_data

        return job_id

    def create_bash_script(self):
        # construct a temporary path for the data to pass it to the bash script
        data_path = f"{self.paths['input']}/{self.name}_data.json"

        # create the data dictionary
        investment_data = {
            "name_experiment": self.name_experiment,
            "exogenous_variables": self.exogenous_variables,
            "endogenous_variables": self.endogenous_variables,
            "general_parameters": self.general_parameters
        }

        # (find better fix) ensure data does not have np.int64 types
        def convert_int64(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_int64(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_int64(v) for v in obj]
            return obj

        investment_data = convert_int64(investment_data)

        # write investment data to a separate JSON file
        with open(data_path, 'w') as f:
            json.dump(investment_data, f)

        # write the bash script
        with open(self.paths['bash_script'], 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=9:00:00
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
END
''')

        return self.paths['bash_script']


if __name__ == '__main__':
    main()
