"""
File name: solver_module.py
Description: this file implements the Solver class.

The Solver class represents a zero-profit condition problem, where you supply
a base xml file, a set of exogenous variables, endogenous variables, and general parameters.

Public methods:
    - solve: solves the investment problem.
    - last_run: returns the equilibrium run if the last iteration converged.
    - submit: submits the investment problem to Quest.
"""
import sys
from pathlib import Path
import logging
import json
import shutil
from pprint import pprint

from .run_module import Run
from .optimization_module import (OptimizationPathEntry,
                                  derivatives_from_profits,
                                  get_last_successful_iteration)
from .utils.auxiliary import submit_slurm_job, wait_for_jobs
from .constants import initialize_paths_investment_problem, create_investment_name

logger = logging.getLogger(__name__)

# In GB
MEMORY_REQUESTED: int = 10
UNSUCCESSFUL_RUN: int = 13


class Solver:
    def __init__(self,
                 parent_folder: str,
                 exogenous_variable: dict,
                 endogenous_variables: dict,
                 general_parameters: dict):
        """
        Initializes the InvestmentProblem object.

        Args:
            - folder: Path to the folder where the investment problem is stored.
            - exogenous_variable: Dictionary containing the exogenous variables.
            - endogenous_variables: Dictionary containing the endogenous variables.
            - general_parameters: Dictionary containing general parameters for the run.
                o all the parameters from the Run class.
                o requested_time_run: Requested time for the run.
        """
        parent_folder: Path = Path(parent_folder)
        self.exogenous_variable: dict = exogenous_variable
        self.endogenous_variables: dict = endogenous_variables
        self.general_parameters: dict = general_parameters

        parent_name: str = parent_folder.name
        self.name: str = create_investment_name(
            parent_name, exogenous_variable)

        # initalize relevant paths
        self.paths: dict[str, Path] = initialize_paths_investment_problem(
            parent_folder, self.name)

        solver_options_default: dict = {
            'max_iter': 80,
            'delta': 10,
            'threshold_profits': 0.01,
        }
        self.solver_options: dict = general_parameters.get(
            'solver_options', solver_options_default)

        # initialize the optimization trajectory
        self.solver_trajectory: list = self._init_trajectory()

    def _init_trajectory(self):
        # Check if the optimization trajectory file exists
        if self.paths['solver_trajectory'].exists():
            # If yes, load it
            with open(self.paths['solver_trajectory'], 'r') as file:
                data = json.load(file)
                solver_trajectory = [OptimizationPathEntry.from_dict(entry)
                                     for entry in data]

            logger.info("Successfully loaded optimization trajectory from %s.",
                        self.paths['solver_trajectory'])
            logger.debug("Optimization trajectory for %s:", self.name)
            pprint(solver_trajectory)

        else:
            # If not, initialize it with the initial guess
            solver_trajectory: list[OptimizationPathEntry] = []
            logger.info(
                "Initializing optimization trajectory at the initial guess.")

            # initialize the first iteration
            iteration_0 = OptimizationPathEntry(
                iteration=0,
                current_investment={
                    endogenous_variable: entry['initial_guess']
                    for endogenous_variable, entry in self.endogenous_variables.items()
                },
                successful=False,
            )
            solver_trajectory.append(iteration_0)

        return solver_trajectory

    def _save_trajectory(self):
        with open(self.paths['solver_trajectory'], 'w') as file:
            json.dump([entry.to_dict() for entry in self.solver_trajectory],
                      file, indent=4, sort_keys=True)
        logger.info('Saved optimization trajectory to %s',
                    self.paths["solver_trajectory"])

    def prototype(self):
        # Create run at the initial guesses
        initial_guess = {
            endogenous_variable: entry['initial_guess']
            for endogenous_variable, entry in self.endogenous_variables.items()
        }
        run_initial_guess = self.create_run(initial_guess)
        run_initial_guess.prototype()

    #########################################
    # Methods to solve the
    def solve(self):
        """
        This function implements the Newton-Raphson algorithm to solve the zero-profit
        condition problem.

        The algorithm is as follows:
        1. Initialize the solver at the initial guess.
        2. Start the Newton-Raphson loop.
        3. Compute the profits and derivatives. This is done by submitting the runs for the 
            current investment_problem and the perturbed investments,
            and extracting the profits from the runs.
        4. Check for convergence. If the profits are within the threshold, the algorithm stops.
        """
        # Get the current iteration
        current_iteration: OptimizationPathEntry = self.solver_trajectory[-1]
        logger.info('Initializing the solver at iteration %s',
                    current_iteration)

        # Check for convergence
        # if current_iteration.check_convergence():
        #    logger.info(
        #        'Convergence reached. Current iteration %s', current_iteration)
        #    return

        # Start the Newton-Raphson loop
        while current_iteration.iteration < self.solver_options['max_iter']:
            logger.info('Current iteration: %s', current_iteration)

            # Compute the profits and derivatives
            current_iteration = self._update_current_iteration(
                current_iteration)

            # Update the optimization trajectory
            self.solver_trajectory[-1] = current_iteration
            self._save_trajectory()

            # Check for convergence
            if current_iteration.check_convergence():
                logger.info(
                    'Convergence reached. Current iteration %s', current_iteration)
                break

            # Compute the new investment
            new_iteration: OptimizationPathEntry = current_iteration.next_iteration()
            logger.info('Next iteration: %s', new_iteration)

            # Append the new iteration to the optimization trajectory
            self.solver_trajectory.append(new_iteration)
            self._save_trajectory()

            # Clear the runs folders, except for the current run
            self.clear_runs_folders(current_iteration.current_investment)

            # Update the current iteration loop variable
            current_iteration: OptimizationPathEntry = new_iteration

        logger.info(
            'Maximum number of iterations reached. Optimization trajectory saved.')

        # save results
        self._save_trajectory()

    def clear_runs_folders(self, current_investment=None, force=False):
        """
        Deletes all the directories in self.paths['folder']
        """
        if current_investment is None:
            current_iteration = get_last_successful_iteration(
                self.solver_trajectory)
            current_investment = current_iteration.current_investment
        perturbed_runs_dict: dict[str, Run] = self.perturbed_runs(
            current_investment)
        if force:
            perturbed_runs_names = [perturbed_runs_dict['current'].name]
        else:
            perturbed_runs_names = [
                run.name for run in perturbed_runs_dict.values()]
        for directory in self.paths['folder'].iterdir():
            # Check if the directory is in the perturbed runs names
            if directory.is_dir() and directory.name not in perturbed_runs_names:
                logger.info("Deleting directory %s", directory)
                try:
                    shutil.rmtree(directory)
                    logger.info("Successfully deleted directory %s", directory)
                except OSError:
                    logger.critical("Could not delete directory %s",
                                    directory)

    def _update_current_iteration(self,
                                  current_iteration: OptimizationPathEntry) -> OptimizationPathEntry:
        '''
        Updates the current iteration with the profits and derivatives
        '''
        # if current_iteration.profits and current_iteration.profits_derivatives:
        #    logger.info('Profits and derivatives already computed for iteration %s',
        #                current_iteration.iteration)
        #    return current_iteration

        logger.info('Preparing to compute profits for iteration with %s',
                    current_iteration.current_investment)

        # Compute the profits and derivatives
        profits, profits_derivatives = self.profits_and_derivatives(
            current_iteration.current_investment)

        # Output a new OptimizationPathEntry
        return OptimizationPathEntry(
            iteration=current_iteration.iteration,
            current_investment=current_iteration.current_investment,
            successful=True,
            profits=profits,
            profits_derivatives=profits_derivatives
        )

    @staticmethod
    def submit_and_wait(runs_dict: dict[str, Run],
                        max_attempts: int = 6) -> None:
        '''
        Submits the runs in the runs_dict and waits for them to finish
        '''
        # Submit the runs, give up after max_attempts
        attempts: int = 0
        while attempts < max_attempts:
            job_ids_list: list[str] = []
            for run in runs_dict.values():
                # Check if the run was successful
                if not run.successful():
                    logger.warning("Run %s not successful", run.name)
                    # If not, submit it and append the job_id to the list
                    job_id = run.submit()
                    if job_id:
                        job_ids_list.append(job_id)
                    if attempts == 0:
                        logger.info("First attempt for %s", run.name)
                    else:
                        logger.warning("RETRYING %s, previous attempts = %s", run.name,
                                       attempts)
            if not job_ids_list:
                return None
            logger.info("Waiting for jobs %s", job_ids_list)
            # Wait for the jobs to finish
            wait_for_jobs(job_ids_list)
            attempts += 1

        if attempts == max_attempts:
            logger.critical(
                "Could not successfully finish all runs after %s attempts", max_attempts)
            sys.exit(UNSUCCESSFUL_RUN)

    def perturbed_runs(self, current_investment: dict) -> dict[str, Run]:
        # Create a dict of the perturbed runs
        perturbed_runs_dict: dict[str, Run] = {
            'current': self.create_run(current_investment)}
        for var in self.endogenous_variables.keys():
            investment = current_investment.copy()
            investment[var] += self.solver_options['delta']
            perturbed_runs_dict[var] = self.create_run(investment)
        return perturbed_runs_dict

    def profits_and_derivatives(self, current_investment: dict) -> tuple[dict, dict]:
        '''
        Computes the profits and derivatives for a given level of investment_problem 
        '''
        # Create the runs for the current and perturbed investments
        perturbed_runs_dict: dict[str, Run] = self.perturbed_runs(
            current_investment)

        # Submit the runs
        self.submit_and_wait(perturbed_runs_dict, max_attempts=6)

        # Check if all runs were successful
        if not all(run.successful() for run in perturbed_runs_dict.values()):
            unsuccessful_runs_names = [
                run.name for run in perturbed_runs_dict.values() if not run.successful()]
            logger.critical(
                "Runs %s not successful, aborting investment solver", unsuccessful_runs_names)
            sys.exit(UNSUCCESSFUL_RUN)

        # Extract the profits from the runs
        profits_perturb = {}
        for resource, run in perturbed_runs_dict.items():
            profits_perturb[resource] = run.get_profits()

        # Compute derivatives from the profits of the perturbed runs
        derivatives = derivatives_from_profits(
            profits_perturb, self.solver_options['delta'], list(self.endogenous_variables.keys()))
        return profits_perturb['current'], derivatives

    def create_run(self, current_investment: dict) -> Run:
        """
        Creates a Run object from a level of current investment_problem in the 
        endogenous capacities
        """
        # create a values array with the same order as the variables
        exogenous_variable_temp = {var_name: variable['value'] for var_name,
                                   variable in self.exogenous_variable.items()}
        variables: dict = {**exogenous_variable_temp, **current_investment}
        # create the Run object
        run: Run = Run(self.paths['folder'],
                       self.general_parameters,
                       variables)
        return run

    def investment_results(self, resubmit: bool = False) -> dict:
        """
        Returns the results of the solver for the last iteration.

        Args:
            resubmit: If True, the run is resubmitted before computing the profits, in 
                case it was not successful.
        """
        # Retrieve the last iteration from the solver trajectory
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.solver_trajectory)

        # Construct the header of the row
        exo_vars: dict = {key: entry['value']
                          for key, entry in self.exogenous_variable.items()}

        investment_results: dict = {
            'name': self.name,
            'iteration': last_iteration.iteration,
            **exo_vars,
            **last_iteration.current_investment
        }

        # Compute the profits
        last_run = self.last_run()
        try:
            profits_dict: dict = last_run.get_profits_data_dict(complete=True)
        except FileNotFoundError:
            if resubmit:
                logger.error(
                    "Run %s not successful, resubmitting and returning empty dict",)
                last_run.submit(force=True)
            else:
                logger.error(
                    "Run %s not successful, returning empty dict",)
            return {}

        logger.debug("profits_dict for %s", self.name)
        pprint(profits_dict)

        # Get the list of participants
        participants: list[str] = list(self.endogenous_variables.keys())

        # Update the profits in the last iteration
        last_iteration.profits = {participant: profits_dict[f'{participant}_normalized_profits']
                                  for participant in participants}

        convergence_reached: bool = last_iteration.check_convergence()

        # Append the data to the investment_results dictionary
        investment_results.update(profits_dict)
        investment_results['convergence_reached'] = convergence_reached

        return investment_results

    def last_run(self):
        # Get the last successful iteration
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.solver_trajectory)
        # Create the run object
        run: Run = self.create_run(last_iteration.current_investment)
        return run

    #########################################
    # Methods to submit the job to Quest
    def submit(self):
        """
        Submits the investment problem to Quest. 
        """
        logger.info("Preparing to run %s on quest",
                    self.name)

        script_path: Path = self._create_bash()
        job_id = submit_slurm_job(script_path, job_name=self.name)
        return job_id

    def _create_bash(self):
        # Convert the investment data to JSON directly
        investment_data = {
            "parent_folder": str(self.paths['parent_folder']),
            "exogenous_variable": self.exogenous_variable,
            "endogenous_variables": self.endogenous_variables,
            "general_parameters": self.general_parameters
        }

        investment_data_str = json.dumps(investment_data)

        requested_time: float = self.general_parameters['slurm']['solver']['time']
        hours = int(requested_time)
        minutes = int((requested_time - hours) * 60)
        seconds = int(((requested_time - hours) * 60 - minutes) * 60)

        logging_level: str = self.general_parameters['slurm']['solver'].get(
            'log_level', 'DEBUG')

        requested_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        email = self.general_parameters['slurm']['solver'].get('email',
                                                               None)
        if email:
            email_line = f"#SBATCH --mail-user={email}"
        else:
            email_line = ""

        # Write the bash script without a separate data file
        with open(self.paths['bash'], 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time_str}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem={MEMORY_REQUESTED}G
#SBATCH --job-name={self.name}
#SBATCH --output={self.paths['slurm_out']}
#SBATCH --error={self.paths['slurm_err']}
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --exclude=qhimem[0207-0208]
{email_line}

module purge
module load python-miniconda3/4.12.0

python - <<END

print('Solving the investment problem {self.name}')

import sys
import json

# Load investment data from inline JSON string
investment_data = json.loads('''{investment_data_str}''')

sys.path.append('/projects/p32342/code')
from src.solver_module import Solver
from src.utils.logging_config import setup_logging
import logging

setup_logging(level=logging.{logging_level})

solver = Solver(**investment_data)
print('Successfully loaded the solvers data')
solver.solve()
END
""")

        return self.paths['bash']
