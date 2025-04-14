"""
Module: data_analysis_module.py
Author: Pedro Bitencourt (Northwestern University)
"""

import pandas as pd
from typing import Dict
import logging


logger = logging.getLogger(__name__)

###########################################################################################
# Functions to compute the objective function of the Solver class


def profits_per_participant(run_df: pd.DataFrame,
                            capacities: dict[str, float],
                            cost_parameters: dict) -> Dict:
    """
    Computes profits for the specified endogenous variables.

    Returns:
        dict: A dictionary of profits.
    """
    participants: list[str] = list(capacities.keys())

    results_dict: dict = {}

    def compute_participant_metrics(run_df: pd.DataFrame, participant: str,
                                    capacity_mw: float) -> dict:
        """
        For a given run, compute the economic metrics for a given participant.

        Arguments:
            run_df (pd.DataFrame): DataFrame containing the data for the run.
            participant (str): Name of the participant.
            capacity_mw (float): Capacity of the participant in MW.
        """
        def revenue_hour(run_df: pd.DataFrame, participant: str) -> float:
            revenues: float = (run_df[f'production_{participant}'] *
                               run_df['marginal_cost']).mean()
            return revenues

        def variable_costs_hour(run_df: pd.DataFrame, participant: str) -> float:
            if participant != 'thermal':
                return 0
            variable_costs: float = (
                run_df[f'variable_cost_{participant}']).mean()
            return variable_costs

        revenue = revenue_hour(run_df, participant)
        variable_costs = variable_costs_hour(run_df, participant)

        fixed_costs = cost_parameters[f"fixed_cost_{participant}"]

        return {
            f'{participant}_capacity_mw': capacity_mw,
            f'{participant}_revenue_hour': revenue,
            f'{participant}_variable_costs_hour': variable_costs,
            f'{participant}_revenue_mw_hour': revenue / capacity_mw,
            f'{participant}_variable_costs_mw_hour': variable_costs / capacity_mw,
            f'{participant}_fixed_costs_mw_hour': fixed_costs,
            f'{participant}_total_costs_mw_hour': (fixed_costs + variable_costs / capacity_mw),
            f'{participant}_profits_mw_hw': (revenue - variable_costs)/capacity_mw - fixed_costs,
            f'{participant}_normalized_profits': (((revenue - variable_costs)/capacity_mw - fixed_costs)
                                                  / (fixed_costs + variable_costs / capacity_mw))
        }

    for participant in participants:
        if participant in ('salto', 'hydro'):
            continue
        capacity: float = capacities[participant]
        results_dict.update(compute_participant_metrics(
            run_df, participant, capacity))

    results_dict['avg_price'] = run_df['marginal_cost'].mean()

    return results_dict
