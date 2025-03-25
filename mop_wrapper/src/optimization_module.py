"""
File name: optimization_module.py
Author: Pedro Bitencourt
Description: this file implements the OptimizationPathEntry class and related methods.
"""
import logging
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .constants import ERROR_CODE_UNSUCCESSFUL_ITERATION

THRESHOLD_PROFITS = 0.005

logger = logging.getLogger(__name__)


@dataclass
class Iteration:
    iteration: int
    capacities: dict[str, float]
    successful: bool
    profits: Optional[dict[str, float]] = None
    profits_derivatives: Optional[dict[str, dict[str, float]]] = None

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'capacities': self.capacities,
            'successful': self.successful,
            'profits': self.profits,
            'profits_derivatives': self.profits_derivatives
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def capacities_array(self) -> np.ndarray:
        return np.array([self.capacities[var] for var in self.capacities.keys()])

    def profits_array(self) -> Optional[np.ndarray]:
        if self.profits is None:
            return None
        return np.array([self.profits[var] for var in self.capacities.keys()])

    def profits_derivatives_array(self) -> Optional[np.ndarray]:
        if self.profits_derivatives is None:
            return None
        return np.array([
            [self.profits_derivatives[row_var][col_var]
                for col_var in self.capacities.keys()]
            for row_var in self.capacities.keys()
        ])

    def check_convergence(self) -> bool:
        if self.profits is None:
            return False
        for var in self.capacities.keys():
            profit = self.profits[var]
            if self.capacities[var] == 1:
                profit = np.maximum(profit, 0)
            if np.abs(profit) >= THRESHOLD_PROFITS:
                return False
        return True

    def next_iteration(self) -> 'Iteration':
        # check if the current iteration was successful
        if not self.successful:
            logger.critical('Iteration %s was not successful before calling next_iteration. Aborting.',
                            self.iteration)
            sys.exit(ERROR_CODE_UNSUCCESSFUL_ITERATION)

        # Transform the dictionaries into np.arrays
        profits_array = self.profits_array()
        profits_derivatives_array = self.profits_derivatives_array()
        capacities_array = self.capacities_array()

        # check if the profits and derivatives are not None
        if profits_array is None or profits_derivatives_array is None:
            logger.error(
                "Profits or derivatives are None. Aborting.")
            sys.exit(1)

        # Compute the new capacities
        new_capacities_array = newton_iteration(
            profits_array, profits_derivatives_array, capacities_array)

        # Adjust the iteration to avoid negative investment
        new_capacities_array = adjust_iteration(new_capacities_array,
                                                capacities_array)

        # Round new_capacities to nearest unit
        new_capacities_array = np.round(new_capacities_array)

        if np.max(new_capacities_array) > 12000:
            logger.critical("New iteration is off bounds: %s",
                            new_capacities_array)
            logger.critical("Current iteration: %s", capacities_array)
            sys.exit(3)

        # Transform the new capacities into a dictionary
        new_capacities_dict: dict[str, float] = {
            var: float(new_capacities_array[i]) for i, var in enumerate(self.capacities.keys())
        }

        # Create and return a new OptimizationPathEntry
        return Iteration(
            iteration=self.iteration + 1,
            capacities=new_capacities_dict,
            successful=False,  # This will be set to True after profits are computed
            profits=None,
            profits_derivatives=None
        )


def adjust_iteration(new_capacities_array, capacities_array):
    # Get rid of negative capacities
    new_capacities_array = np.maximum(new_capacities_array, 1)

    # To avoid the boundaries
    updated_capacities_array = 0.5*(new_capacities_array +
                                    capacities_array)
    return updated_capacities_array


def newton_iteration(profits_array: np.ndarray,
                     profits_derivatives_array: np.ndarray,
                     capacities_array: np.ndarray) -> np.ndarray:
    # Compute the new capacities using numpy operations
    try:
        capacities_change = np.linalg.solve(
            profits_derivatives_array, profits_array)
        new_capacities_array = capacities_array - capacities_change
    except np.linalg.LinAlgError:
        # Handle singular matrix error
        logger.critical("Jacobian matrix: %s", profits_derivatives_array)
        logger.critical(
            "Singular matrix encountered. Aborting.")
        sys.exit(1)

    return new_capacities_array


def derivatives_from_profits(profits: dict, delta: float, endogenous_variables: list[str]):
    logger.debug("profits dict %s", profits)
    logger.debug("endogenous variables dict %s", endogenous_variables)
    derivatives = {}
    for r1 in endogenous_variables:
        derivatives[r1] = {}
        for r2 in endogenous_variables:
            profit_change = profits[r2][r1] - profits['current'][r1]
            derivatives[r1][r2] = profit_change / delta
    return derivatives


def get_last_successful_iteration(trajectory: list[Iteration]) -> Iteration:
    for entry in reversed(trajectory):
        if entry.successful:
            return entry
    logger.warning(
        "No successful iteration found in the optimization trajectory, returning the first entry.")
    logger.debug("Optimization trajectory: %s", trajectory)
    return trajectory[0]
