"""
File name: investment_module.py
Description: this file implements the InvestmentProblem class.

The InvestmentProblem class represents a zero-profit condition problem, where you supply
a base xml file, a set of exogenous variables, endogenous variables, and general parameters. 

Public methods:
    - __init__: initializes the InvestmentProblem object.
    - __repr__: returns a string representation of the object.
    - solve_investment: solves the investment problem.
    - equilibrium_run: returns the equilibrium run if the last iteration converged.
    - submit: submits the investment problem to Quest.
"""
import sys
from pathlib import Path
import logging
import json
import numpy as np
import shutil

from src.run_module import Run
from src.run_processor_module import RunProcessor
from src.optimization_module import (OptimizationPathEntry,
                                     derivatives_from_profits,
                                     print_optimization_trajectory_function,
                                     get_last_successful_iteration)
from src.auxiliary import submit_slurm_job, wait_for_jobs, make_name
from src.constants import DELTA, MAX_ITER, UNSUCCESSFUL_RUN


logger = logging.getLogger(__name__)


class InvestmentProblem:
    def __init__(self,
                 parent_folder: str,
                 exogenous_variables: dict,
                 endogenous_variables: dict,
                 general_parameters: dict):
        """
        Initializes the InvestmentProblem object.

        Args:
            - folder: Path to the folder where the investment problem is stored.
            - exogenous_variables: Dictionary containing the exogenous variables.
            - endogenous_variables: Dictionary containing the endogenous variables.
            - general_parameters: Dictionary containing general parameters for the run.
                o all the parameters from the Run class.
                o requested_time_run: Requested time for the run.
        """
        parent_folder: Path = Path(parent_folder)
        self.exogenous_variables: dict = exogenous_variables
        self.endogenous_variables: dict = endogenous_variables
        self.general_parameters: dict = general_parameters

        self.name: str = self._create_name(parent_folder)
        logging.info("Initializing investment problem %s", self.name)

        # initalize relevant paths
        self.paths: dict[str, Path] = self._initialize_paths(parent_folder)

        # initialize the optimization trajectory
        self.optimization_trajectory: list = self._initialize_optimization_trajectory()

    def _create_name(self, parent_folder: Path):
        exog_var_values: list[float] = [variable['value'] for variable in
                                        self.exogenous_variables.values()]
        parent_name: str = parent_folder.name
        name: str = f"{parent_name}_investment_{make_name(exog_var_values)}"
        return name

    def __repr__(self):
        return f"InvestmentProblem: exogenous variables {self.exogenous_variables}"

    def _initialize_paths(self, folder: Path):
        paths: dict[str, Path] = {}
        paths['parent_folder'] = folder
        paths['folder'] = folder / self.name
        paths['bash'] = folder / self.name / f'{self.name}.sh'
        paths['optimization_trajectory'] = folder / \
            f'{self.name}_trajectory.json'

        # create the directory
        paths['folder'].mkdir(parents=True, exist_ok=True)

        return paths

    def _initialize_optimization_trajectory(self):
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
                    endogenous_variable: entry['initial_guess']
                    for endogenous_variable, entry in self.endogenous_variables.items()
                },
                endogenous_variables=list(self.endogenous_variables.keys()),
                successful=False,
            )
            optimization_trajectory.append(iteration_0)
        return optimization_trajectory

    def _save_optimization_trajectory(self):
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
            current_iteration = self._update_current_iteration(
                current_iteration)

            # update the optimization trajectory
            self.optimization_trajectory[-1] = current_iteration

            # save the optimization trajectory
            self._save_optimization_trajectory()

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

            # clear the runs folders
            # self._clear_runs_folders()

        logger.info(
            'Maximum number of iterations reached. Optimization trajectory saved.')

        # save results
        self._save_optimization_trajectory()

    def _clear_runs_folders(self):
        """
        Deletes all the directories in self.paths['folder']
        """
        for directory in self.paths['folder'].iterdir():
            if directory.is_dir():
                shutil.rmtree(directory)

    def _update_current_iteration(self,
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
            profits, profits_derivatives = self._profits_and_derivatives(
                current_investment)

        # if the profits are already computed, use them
        else:
            profits = current_iteration.profits
            profits_derivatives = current_iteration.profits_derivatives

        # output a new OptimizationPathEntry
        return OptimizationPathEntry(
            iteration=current_iteration.iteration,
            current_investment=current_investment,
            endogenous_variables=current_iteration.endogenous_variables,
            successful=True,
            profits=profits,
            profits_derivatives=profits_derivatives
        )

    def _profits_and_derivatives(self, current_investment: dict) -> tuple[dict, dict]:
        '''
        Computes the profits and derivatives for a given level of investment_problem 
        '''
        def create_and_submit_run(investment):
            run = self._create_run(investment)
            return run, run.submit()

        # create dict of investments
        investments_dict = {'current': current_investment}
        for var in self.endogenous_variables.keys():
            investment = current_investment.copy()
            investment[var] += DELTA
            investments_dict[var] = investment

        # create and submit runs
        runs_and_jobs = {
            resource: create_and_submit_run(investment)
            for resource, investment in investments_dict.items()
        }

        # wait for all jobs to complete
        wait_for_jobs([job for _, job in runs_and_jobs.values()])

        # process results
        profits_perturb = {}
        for resource, (run, _) in runs_and_jobs.items():
            run_processor = RunProcessor(run)
            # check if run was successful
            if run_processor:
                endogenous_variables_list: list[str] = list(
                    self.endogenous_variables.keys())
                profits_perturb[resource] = run_processor.get_profits(
                    endogenous_variables_list)
            # if not, abort
            else:
                logging.critical(
                    "Run %s was not successful. Aborting.", run.name)
                sys.exit(UNSUCCESSFUL_RUN)

        # compute derivatives from the profits
        derivatives = derivatives_from_profits(
            profits_perturb, DELTA, list(self.endogenous_variables.keys()))
        return profits_perturb['current'], derivatives

    def _create_run(self, current_investment: dict):
        """
        Creates a Run object from a level of current investment_problem in the 
        endogenous capacities
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
        run: Run = Run(self.paths['folder'],
                       self.general_parameters,
                       variables)

        return run

    def print_optimization_trajectory(self):
        print_optimization_trajectory_function(self.optimization_trajectory)

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
            run: Run = self._create_run(last_iteration.current_investment)
            return run
        return None

    #########################################
    # Methods to submit the job to Quest
    def submit(self):
        """
        Submits the investment problem to Quest. 
        """
        logging.info("Preparing to run %s on quest",
                     self.name)

        script_path: Path = self._create_bash()
        job_id = submit_slurm_job(script_path)

        return job_id

    def _create_bash(self):
        # construct a temporary path for the data to pass it to the bash script
        data_path = f"{self.paths['folder']}/{self.name}_data.json"

        # create the data dictionary
        investment_data = {
            "parent_folder": str(self.paths['parent_folder']),
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
        with open(self.paths['bash'], 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --job-name={self.name}
#SBATCH --output={self.paths['parent_folder']}/{self.name}.out
#SBATCH --error={self.paths['parent_folder']}/{self.name}.err
#SBATCH --mail-user={self.general_parameters['email']}
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0


python - <<END

print('Solving the investment_problem problem {self.name}')

import sys
import json
sys.path.append('/projects/p32342/code')
from src.investment_module import InvestmentProblem

with open('{data_path}', 'r') as f:
    investment_data = json.load(f)
investment_problem = InvestmentProblem(**investment_data)
print('Successfully loaded the investments problem data')
sys.stdout.flush()
sys.stderr.flush()
investment_problem.solve_investment()
END
''')

        return self.paths['bash']
