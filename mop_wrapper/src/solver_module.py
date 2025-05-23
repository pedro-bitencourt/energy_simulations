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
from dataclasses import replace

from .run_module import Run, submit_and_wait_for_runs
from .optimization_module import (Iteration,
                                  derivatives_from_profits,
                                  get_last_successful_iteration)
from .utils.slurm_utils import submit_slurm_job, slurm_header
from .utils.auxiliary import make_name
from .constants import initialize_paths_solver

logger = logging.getLogger(__name__)

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
        self.name: str = create_solver_name(
            parent_name, exogenous_variable)
        self.paths: dict[str, Path] = initialize_paths_solver(
            parent_folder, self.name)
        self.folder: Path = self.paths['folder']

        solver_options_default: dict = {
            'max_iter': 200,
            'delta': 10,
            'threshold_profits': 0.01,
        }
        self.solver_options: dict = general_parameters.get(
            'solver_options', solver_options_default)

        self.solver_trajectory: list = self._init_trajectory()

    def _init_trajectory(self):
        # Check if the optimization trajectory file exists
        if self.paths['solver_trajectory'].exists():
            # If yes, load it
            with open(self.paths['solver_trajectory'], 'r') as file:
                data = json.load(file)
                solver_trajectory = [Iteration.from_dict(entry)
                                     for entry in data]
            logger.info("Successfully loaded optimization trajectory from %s.",
                        self.paths['solver_trajectory'])
        else:
            # Create the path for the solver_trajectory
            self.paths['solver_trajectory'].parent.mkdir(parents=True,
                                                         exist_ok=True)
            # If not, initialize it with the initial guess
            solver_trajectory: list[Iteration] = []
            logger.info(
                "Initializing optimization trajectory at the initial guess.")

            # initialize the first iteration
            iteration_0 = Iteration(
                iteration=0,
                capacities={
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
        run_initial_guess = self.run_factory(initial_guess)
        run_initial_guess.prototype()
        return run_initial_guess

    def run_factory(self, capacities: dict) -> Run:
        """
        Creates a Run object from a level of current investment_problem in the 
        endogenous capacities
        """
        # create a values array with the same order as the variables
        exogenous_variable_temp = {var_name: variable['value'] for var_name,
                                   variable in self.exogenous_variable.items()}
        variables: dict = {**exogenous_variable_temp, **capacities}
        # create the Run object
        logger.debug("Creating run with variables: %s", variables)
        run: Run = Run(self.paths['folder'],
                       self.general_parameters,
                       variables)
        return run

    #########################################
    # Methods to solve the competitive equilibrium
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
        current_iteration: Iteration = self.solver_trajectory[-1]
        logger.info('Initializing the solver at iteration %s',
                    current_iteration)

        # Start the Newton-Raphson loop
        while current_iteration.iteration < self.solver_options['max_iter']:
            logger.info('Current iteration: %s', current_iteration)

            # Compute the current iteration
            profits, profits_derivatives = self.profits_and_derivatives(
                current_iteration.capacities)
            current_iteration = replace(current_iteration, successful=True,
                                        profits=profits, profits_derivatives=profits_derivatives)

            # Update the optimization trajectory
            self.solver_trajectory[-1] = current_iteration
            self._save_trajectory()

            # Check for convergence
            if current_iteration.check_convergence():
                logger.info(
                    'Convergence reached. Current iteration %s', current_iteration)

                self.save_solver_results()
                return

            # Compute the new investment
            new_iteration: Iteration = current_iteration.next_iteration()
            logger.info('Next iteration: %s', new_iteration)

            # Append the new iteration to the optimization trajectory
            self.solver_trajectory.append(new_iteration)
            self._save_trajectory()

            # Clear the runs folders, except for the current run
            self.clear_runs_folders()

            # Update the current iteration loop variable
            current_iteration: Iteration = new_iteration

        logger.info(
            'Maximum number of iterations reached. Optimization trajectory saved.')

        # save results
        self._save_trajectory()
        self.save_solver_results()
        return

    def save_solver_results(self):
        json.dump(self.solver_results(), open(
            self.paths['solver_results'], 'w'), indent=4)

    def last_capacities(self):
        return get_last_successful_iteration(self.solver_trajectory).capacities

    def clear_runs_folders(self):
        last_run: Run = self.last_run()
        for directory in self.paths['folder'].iterdir():
            if directory.is_dir() and directory != last_run.folder:
                logger.info("Deleting directory %s", directory)
                try:
                    shutil.rmtree(directory)
                    logger.info("Successfully deleted directory %s", directory)
                except OSError:
                    logger.warning("Could not delete directory %s",
                                   directory)

    def redo_run(self, delete: bool = False):
        run = self.last_run()
        logger.info("Resubmitting run %s", run.name)
        if delete:
            logger.info("Deleting run %s", run.name)
            run.delete()
            run.submit()

    def perturbed_runs(self, capacities: dict) -> dict[str, Run]:
        # Create a dict of the perturbed runs to compute the derivatives
        perturbed_runs_dict: dict[str, Run] = {
            'current': self.run_factory(capacities)}
        for var in self.endogenous_variables.keys():
            investment = capacities.copy()
            investment[var] += self.solver_options['delta']
            perturbed_runs_dict[var] = self.run_factory(investment)
        return perturbed_runs_dict

    def profits_and_derivatives(self, capacities: dict) -> tuple[dict, dict]:
        '''
        Computes the profits and derivatives for a given level of investment_problem 
        '''
        # Create the runs for the current and perturbed investments
        perturbed_runs_dict: dict[str, Run] = self.perturbed_runs(
            capacities)

        # Submit the runs
        submit_and_wait_for_runs(perturbed_runs_dict, max_attempts=6)

        # Check if all runs were successful
        if not all(run.successful() for run in perturbed_runs_dict.values()):
            unsuccessful_runs_names = [
                run.name for run in perturbed_runs_dict.values() if not run.successful()]
            logger.critical(
                "Runs %s not successful, aborting investment solver", unsuccessful_runs_names)
            sys.exit()

        # Extract the profits from the runs
        profits_perturb = {}
        for resource, run in perturbed_runs_dict.items():
            profits_perturb[resource] = run.get_profits()

        # Compute derivatives from the profits of the perturbed runs
        derivatives = derivatives_from_profits(
            profits_perturb, self.solver_options['delta'], list(self.endogenous_variables.keys()))
        return profits_perturb['current'], derivatives

    def solver_results(self, resubmit: bool = False) -> dict:
        """
        Returns the results of the solver for the last iteration.

        Args:
            resubmit: If True, the run is resubmitted before computing the profits, in 
                case it was not successful.
        """
        last_iteration: Iteration = get_last_successful_iteration(
            self.solver_trajectory)
        convergence_reached: bool = last_iteration.check_convergence()
        if not convergence_reached:
            logger.error(
                "Convergence not reached, current iteration: %s", last_iteration)
            if resubmit:
                logger.info("Resubmitting the solver")
                self.submit()

        last_run: Run = self.last_run()

        def create_header(last_iteration: Iteration) -> dict:
            exo_vars: dict = {key: entry['value']
                              for key, entry in self.exogenous_variable.items()}
            profits_dict: dict = {f"profits_{key}": value for key, value in
                                  last_iteration.profits.items()}
            header: dict = {
                'name': self.name,
                'iteration': last_iteration.iteration,
                **profits_dict,
                **exo_vars,
                **last_iteration.capacities
            }
            return header

        solver_results: dict = create_header(last_iteration)
        solver_results['convergence_reached'] = convergence_reached

        return solver_results

    def last_run(self):
        # Get the last successful iteration
        last_iteration: Iteration = get_last_successful_iteration(
            self.solver_trajectory)
        # Create the run object
        run: Run = self.run_factory(last_iteration.capacities)
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

    def serialize(self):
        investment_data = {
            "parent_folder": str(self.paths['parent_folder']),
            "exogenous_variable": self.exogenous_variable,
            "endogenous_variables": self.endogenous_variables,
            "general_parameters": self.general_parameters
        }
        return json.dumps(investment_data)

    def _create_bash(self):
        investment_data_str = self.serialize()

        header = slurm_header(
            slurm_configs=self.general_parameters.get('slurm', None),
            job_name=self.name,
            job_type="solver",
            slurm_path=self.paths['bash'],
            # email=self.general_parameters['slurm'].get('email')
            email=self.general_parameters.get('email')
        )

        logging_level: str = self.general_parameters.get(
            'log_level', 'INFO')

        # Write the bash script without a separate data file
        with open(self.paths['bash'], 'w') as f:
            f.write(f"""{header}

module purge
module load python-miniconda3/4.12.0

python - <<END

print('Solving the investment problem {self.name}')

import sys
import json

# Load investment data from inline JSON string
investment_data = json.loads('''{investment_data_str}''')

sys.path.append('/projects/p32342/code/mop_wrapper')
from src.solver_module import Solver
from src.utils.logging_config import setup_logging

setup_logging(level="{logging_level}")

solver = Solver(**investment_data)
print('Successfully loaded the solvers data')
solver.solve()
END
""")
        return self.paths['bash']


def create_solver_name(parent_name: str, exogenous_variable: dict) -> str:
    exog_var_value: list[float] = [variable['value'] for variable in
                                   exogenous_variable.values()]
    print(exog_var_value)
    name: str = make_name(exog_var_value)
    name = f'{parent_name}_{name}'
    return name
