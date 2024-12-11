"""
File name: investment_module.py
Description: this file implements the InvestmentProblem class.

The InvestmentProblem class represents a zero-profit condition problem, where you supply
a base xml file, a set of exogenous variables, endogenous variables, and general parameters.

Public methods:
    - __init__: initializes the InvestmentProblem object.
    - __repr__: returns a string representation of the object.
    - solve_investment: solves the investment problem.
    - last_run: returns the equilibrium run if the last iteration converged.
    - submit: submits the investment problem to Quest.
"""
import sys
from pathlib import Path
import logging
import json
from typing import Optional
import shutil
from pprint import pprint

from src.run_module import MEMORY_REQUESTED, Run
from src.run_processor_module import RunProcessor
from src.optimization_module import (OptimizationPathEntry,
                                     derivatives_from_profits,
                                     print_optimization_trajectory_function,
                                     get_last_successful_iteration)
from src.auxiliary import submit_slurm_job, wait_for_jobs
from src.constants import DELTA, MAX_ITER, UNSUCCESSFUL_RUN, initialize_paths_investment_problem, create_investment_name

logger = logging.getLogger(__name__)

# In GB
MEMORY_REQUESTED: int = 10


class InvestmentProblem:
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

        self.name: str = self._create_name(parent_folder)
        logger.info("Initializing investment problem %s", self.name)

        # initalize relevant paths
        self.paths: dict[str, Path] = initialize_paths_investment_problem(
            parent_folder, self.name)

        # initialize the optimization trajectory
        self.optimization_trajectory: list = self._initialize_optimization_trajectory()


    def _initialize_optimization_trajectory(self):
        # Check if the optimization trajectory file exists
        if self.paths['optimization_trajectory'].exists():
            # If yes, load it
            with open(self.paths['optimization_trajectory'], 'r') as file:
                data = json.load(file)
                optimization_trajectory = [OptimizationPathEntry.from_dict(entry)
                                           for entry in data]

            logger.info("Successfully loaded optimization trajectory from %s.",
                        self.paths['optimization_trajectory'])
            logger.debug("Optimization trajectory for %s:", self.name)
            pprint(optimization_trajectory)

        else:
            # If not, initialize it with the initial guess
            optimization_trajectory: list[OptimizationPathEntry] = []
            logger.critical("Optimization trajectory not found at %s. Initializing a new one.",
                            self.paths['optimization_trajectory'])

            # initialize the first iteration
            iteration_0 = OptimizationPathEntry(
                iteration=0,
                current_investment={
                    endogenous_variable: entry['initial_guess']
                    for endogenous_variable, entry in self.endogenous_variables.items()
                },
                successful=False,
            )
            optimization_trajectory.append(iteration_0)

        return optimization_trajectory

    def _save_optimization_trajectory(self):
        with open(self.paths['optimization_trajectory'], 'w') as file:
            json.dump([entry.to_dict() for entry in self.optimization_trajectory],
                      file, indent=4, sort_keys=True)
        logger.info('Saved optimization trajectory to %s',
                    self.paths["optimization_trajectory"])

    def prototype(self):
        # Create run at the initial guesses
        initial_guess = {
            endogenous_variable: entry['initial_guess']
            for endogenous_variable, entry in self.endogenous_variables.items()
        }
        run_initial_guess = self.create_run(initial_guess)
        run_initial_guess.prototype()

    def solve_investment(self):
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
        current_iteration: OptimizationPathEntry = self.optimization_trajectory[-1]
        logger.info('Initializing the solver at iteration %s',
                    current_iteration)

        # IF the force flag is set, compute the profits for the current investment
        if self.general_parameters.get('force', False):
            run: Run = self.create_run(
                current_iteration.current_investment)
            if run.successful():
                profits = RunProcessor(run).get_profits()
                current_iteration.profits = profits

        # Check for convergence
        if current_iteration.check_convergence():
            logger.info(
                'Convergence reached. Current iteration %s', current_iteration)
            return

        # Start the Newton-Raphson loop
        while current_iteration.iteration < MAX_ITER:
            logger.info('Current iteration: %s', current_iteration)

            # Compute the profits and derivatives
            current_iteration = self._update_current_iteration(
                current_iteration)

            # Update the optimization trajectory
            self.optimization_trajectory[-1] = current_iteration

            # Save the optimization trajectory
            self._save_optimization_trajectory()

            # Check for convergence
            if current_iteration.check_convergence():
                logger.info(
                    'Convergence reached. Current iteration %s', current_iteration)
                break

            # Compute the new investment
            new_iteration: OptimizationPathEntry = current_iteration.next_iteration()

            logger.info('Next iteration: %s', new_iteration)

            # Update the current iteration loop variable
            current_iteration: OptimizationPathEntry = new_iteration

            # Append the new iteration to the optimization trajectory
            self.optimization_trajectory.append(current_iteration)

            # Clear the runs folders, except for the current run
            current_run: Run = self.create_run(
                current_iteration.current_investment)
            self.clear_runs_folders(current_run.name)

        logger.info(
            'Maximum number of iterations reached. Optimization trajectory saved.')

        # save results
        self._save_optimization_trajectory()

    def clear_runs_folders(self, last_run_name: str):
        """
        Deletes all the directories in self.paths['folder']
        """
        for directory in self.paths['folder'].iterdir():
            if directory.is_dir() and directory.name != last_run_name:
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
        # Check if the current iteration was successful
        if current_iteration.profits is None or current_iteration.profits_derivatives is None:
            logger.info('Preparing to compute profits for iteration with %s',
                        current_iteration.current_investment)

            # Compute the profits and derivatives
            profits, profits_derivatives = self.profits_and_derivatives(
                current_iteration.current_investment)

        # If the profits are already computed, use them
        else:
            profits = current_iteration.profits
            profits_derivatives = current_iteration.profits_derivatives

        # Output a new OptimizationPathEntry
        return OptimizationPathEntry(
            iteration=current_iteration.iteration,
            current_investment=current_iteration.current_investment,
            successful=True,
            profits=profits,
            profits_derivatives=profits_derivatives
        )

    def submit_and_wait(self, runs_dict: dict[str, Run], 
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
                successful: bool = run.successful()
                # If not, submit it and append the job_id to the list
                if not successful:
                    job_id = run.submit()
                    if job_id:
                        job_ids_list.append(job_id)
                if attempts == 1:
                    logger.info("First attempt for %s", run.name)
                else:
                    logger.critical("RETRYING %s, attempts = %s", run.name,
                                    attempts)
            # Wait for the jobs to finish
            wait_for_jobs(job_ids_list)
            attempts += 1


    def profits_and_derivatives(self, current_investment: dict) -> tuple[dict, dict]:
        '''
        Computes the profits and derivatives for a given level of investment_problem 
        '''
        # Create a dict of the perturbed runs
        perturbed_runs_dict: dict[str, Run] = {
            'current': self.create_run(current_investment)}
        for var in self.endogenous_variables.keys():
            investment = current_investment.copy()
            investment[var] += DELTA
            perturbed_runs_dict[var] = self.create_run(investment)

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
            run_processor = RunProcessor(run)
            profits_perturb[resource] = run_processor.get_profits()

        # Compute derivatives from the profits of the perturbed runs
        derivatives = derivatives_from_profits(
            profits_perturb, DELTA, list(self.endogenous_variables.keys()))
        return profits_perturb['current'], derivatives

    def create_run(self, current_investment: dict) -> Run:
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
        variables: dict = self.exogenous_variable.copy()
        variables.update(endogenous_variables_temp)

        # create the Run object
        run: Run = Run(self.paths['folder'],
                       self.general_parameters,
                       variables)

        return run

    def print_optimization_trajectory(self):
        print_optimization_trajectory_function(self.optimization_trajectory)


    def investment_results(self, lazy: bool =True, resubmit: bool = False) -> dict:
        """
        Returns the results of the investment problem for the last iteration.

        Args:
            lazy: If True, the profits are not computed and the results are returned
                as they are stored in the optimization trajectory.
            resubmit: If True, the run is resubmitted before computing the profits, in 
                case it was not successful.
        """
        # Retrieve the last iteration from the solver trajectory
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.optimization_trajectory)

        # Construct the header of the row
        exo_vars: dict = {key: entry['value']
                          for key, entry in self.exogenous_variable.items()}

        investment_results: dict = {
            'name': self.name,
            'iteration': last_iteration.iteration,
            **exo_vars,
            **last_iteration.current_investment
        }

        # Get the list of participants
        participants: list[str] = list(self.endogenous_variables.keys())

        # Compute the profits
        if lazy:
            profits_dict: Optional[dict] = last_iteration.profits
        else:
            last_run = self.create_run(
                last_iteration.current_investment)
            try:
                last_run_processor = RunProcessor(last_run)
            except FileNotFoundError:
                logger.critical("Run %s not successful, resubmitting and returning empty dict",)
                if resubmit:
                    last_run.submit(force=True)
                return {}
            profits_dict: dict = last_run_processor.profits_data_dict()

            # Update the profits in the last iteration
            last_iteration.profits = {participant: profits_dict[f'{participant}_normalized_profit']
                                      for participant in participants}

        convergence_reached: bool = last_iteration.check_convergence()

        # Append the data to the investment_results dictionary
        investment_results.update(profits_dict)
        investment_results['convergence_reached'] = convergence_reached
        return investment_results

    def last_run(self):
        """
        Returns the equilibrium run if the last iteration converged
        """
        # Get the last successful iteration
        last_iteration: OptimizationPathEntry = get_last_successful_iteration(
            self.optimization_trajectory)

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
        job_id = submit_slurm_job(script_path)
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
    
        requested_time: float = self.general_parameters['requested_time_solver']
        hours = int(requested_time / 3600)
        minutes = int((requested_time - hours * 3600) / 60)
        seconds = int((requested_time - hours * 3600) % 60)

        requested_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
        if self.general_parameters.get('email', None):
            email_line = f"#SBATCH --mail-user={self.general_parameters['email']}"
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
#SBATCH --output={self.paths['parent_folder']}/{self.name}.out
#SBATCH --error={self.paths['parent_folder']}/{self.name}.err
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
from src.investment_module import InvestmentProblem

investment_problem = InvestmentProblem(**investment_data)
print('Successfully loaded the investment problem data')
investment_problem.solve_investment()
END
""")
    
        return self.paths['bash']
