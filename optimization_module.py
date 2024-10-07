# Description: Module containing the classes and functions for the optimization process
import logging
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np

ERROR_CODE_UNSUCCESSFUL_ITERATION: int = 3

logger = logging.getLogger(__name__)

THRESHOLD_WIND: int = 4_000
THRESHOLD_SOLAR: int = 4_000


@dataclass
class OptimizationPathEntry:
    iteration: int
    current_investment: dict[str, float]
    successful: bool
    profits: Optional[dict[str, float]] = None
    profits_derivatives: Optional[dict[str, dict[str, float]]] = None

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

    def current_investment_array(self) -> np.ndarray:
        return np.array([self.current_investment['wind'], self.current_investment['solar']])

    def profits_array(self) -> Optional[np.ndarray]:
        if self.profits is None:
            return None
        return np.array([self.profits['wind'], self.profits['solar']])

    def profits_derivatives_array(self) -> Optional[np.ndarray]:
        if self.profits_derivatives is None:
            return None
        return np.array([
            [self.profits_derivatives['wind']['wind'],
                self.profits_derivatives['wind']['solar']],
            [self.profits_derivatives['solar']['wind'],
                self.profits_derivatives['solar']['solar']]
        ])

    def check_convergence(self) -> bool:
        profits = self.profits
        if profits is None:
            return False

        curr_prof_wind = profits['wind']
        curr_prof_solar = profits['solar']

        if self.current_investment['solar'] == 1:
            curr_prof_solar = np.maximum(curr_prof_solar, 0)

        if self.current_investment['wind'] == 1:
            curr_prof_wind = np.maximum(curr_prof_wind, 0)

        curr_prof_wind = np.abs(curr_prof_wind)
        curr_prof_solar = np.abs(curr_prof_wind)

        return curr_prof_wind < THRESHOLD_WIND and curr_prof_solar < THRESHOLD_SOLAR

    def next_iteration(self) -> 'OptimizationPathEntry':
        # check if the current iteration was Successfully
        if not self.successful:
            logger.critical('Iteration %s was not successful before calling next_iteration. Aborting.',
                            self.iteration)
            sys.exit(ERROR_CODE_UNSUCCESSFUL_ITERATION)

        # Transform the dictionaries into np.arrays
        profits_array = self.profits_array()
        profits_derivatives_array = self.profits_derivatives_array()
        current_investment_array = self.current_investment_array()

        # check if the profits and derivatives are not None
        if profits_array is None or profits_derivatives_array is None:
            logger.error(
                "Profits or derivatives are None. Aborting.")
            sys.exit(1)

        # Compute the new investment
        new_investment_array = newton_iteration(
            profits_array, profits_derivatives_array, current_investment_array)

        # Round new_investment to nearest 10
        new_investment_array = np.round(new_investment_array, -1)

        # Get rid of negative investment
        new_investment_array = np.maximum(new_investment_array, 1)

        # Transform the new investment into a dictionary
        new_investment_dict: dict[str, float] = {
            'wind': float(new_investment_array[0]),
            'solar': float(new_investment_array[1])
        }

        # Create and return a new OptimizationPathEntry
        return OptimizationPathEntry(
            iteration=self.iteration + 1,
            current_investment=new_investment_dict,
            successful=False,  # This will be set to True after profits are computed
            profits=None,
            profits_derivatives=None
        )


def newton_iteration(profits_array: np.ndarray,
                     profits_derivatives_array: np.ndarray,
                     current_investment_array: np.ndarray) -> np.ndarray:
    # Compute the new investment using numpy operations
    try:
        investment_change = np.linalg.solve(
            profits_derivatives_array, profits_array)
        new_investment_array = current_investment_array - investment_change
    except np.linalg.LinAlgError:
        # Handle singular matrix error
        logger.critical(
            "Singular matrix encountered. Aborting.")
        sys.exit(1)

    return new_investment_array


def derivatives_from_profits(profits: dict, delta: float):
    # convention:
    # profits['current'][r1] = profit from current investment in r1
    renewables = ['wind', 'solar']
    derivatives = {}
    for r1 in renewables:
        derivatives[r1] = {}
        for r2 in renewables:
            profit_change = profits[r2][r1] - profits['current'][r1]
            # convention: derivatives[r1][r2] = derivative of profit from r1 with respect to investment in r2
            derivatives[r1][r2] = profit_change / delta
    return derivatives


def get_last_successful_iteration(opt_trajectory: list[OptimizationPathEntry]) -> OptimizationPathEntry:
    for entry in reversed(opt_trajectory):
        if entry.successful:
            return entry
    logger.error("""No successful iteration found in the optimization trajectory.
        Returning the first entry.""")
    return opt_trajectory[0]


def print_optimization_trajectory_function(opt_trajectory: list[OptimizationPathEntry]):
    # print the objective function values at the end of each iterations
    obj_fun_values: list = [(entry.iteration, entry.profits)
                            for entry in opt_trajectory]
    investment_path: list = [(entry.iteration, entry.current_investment)
                             for entry in opt_trajectory]
    print("Objective function values:")
    for iteration, profits in obj_fun_values:
        print(f"Iteration {iteration}: {profits}\n")
    print("Investment path:")
    for iteration, investment in investment_path:
        print(f"Iteration {iteration}: {investment}\n")
