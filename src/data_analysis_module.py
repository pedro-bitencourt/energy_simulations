"""
Module: data_analysis_module.py
Author: Pedro Bitencourt (Northwestern University)

Description: this module implements functions to perform data analysis of the dataframes 
    produced by the RunProcessor module. It centralizes all data operations that are not 
    parsing or processing, such as the computation of profits, revenues, prices, etc.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import logging

from .utils.load_configs import load_events


logger = logging.getLogger(__name__)

##################################################################
# Functions to compute the objective function of the Solver class
def profits_per_participant(run_df: pd.DataFrame,
                            capacities: dict[str, float],
                            hourly_fixed_costs: dict) -> Dict:
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
            # Get present value of revenues
            revenues: float = (run_df[f'production_{participant}'] *
                               run_df['marginal_cost']).mean()
            return revenues

        def variable_costs_hour(run_df: pd.DataFrame, participant: str) -> float:
            if participant != 'thermal':
                return 0
            # Get present value of variable costs
            variable_costs: float = (
                run_df[f'variable_cost_{participant}']).mean()
            return variable_costs

        revenue = revenue_hour(run_df, participant)
        variable_costs = variable_costs_hour(run_df, participant)

        fixed_costs = hourly_fixed_costs[participant]
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

    # Update the results dictionary with metrics for each participant
    for participant in participants:
        if participant == 'salto':
            continue
        capacity: float = capacities[participant]
        results_dict.update(compute_participant_metrics(
            run_df, participant, capacity))

    results_dict['avg_price'] = run_df['marginal_cost'].mean()

#    system_total_cost: float = sum(
#        [results_dict[f'{participant}_total_cost'] for participant in participants])
#    system_fixed_cost: float = sum(
#        [results_dict[f'{participant}_fixed_cost'] for participant in participants])
#    results_dict.update({
#        'system_total_cost': system_total_cost,
#        'system_fixed_cost': system_fixed_cost,
#    })

    return results_dict

##################################################################
# Functions to compute results
def full_run_df(run_df: pd.DataFrame, participants: list[str]) -> pd.DataFrame:
    for participant in participants:
        run_df[f'revenue_{participant}'] = (run_df[f'production_{participant}'] *
                                            run_df['marginal_cost'])
        run_df[f'profit_{participant}'] = run_df[f'revenue_{participant}'] - \
            run_df[f'variable_cost_{participant}']

    run_df['production_total'] = run_df[[
        f'production_{participant}' for participant in participants]].sum(axis=1)
    run_df['lost_load'] = (run_df['demand'] -
                           run_df['production_total']).clip(lower=0)
    return run_df

# Helper function to compute metrics for each participant
def conditional_means(run_df: pd.DataFrame, participants: list[str]) -> dict:
    variables = [
        *[f'production_{participant}' for participant in participants],
        *[f'variable_cost_{participant}' for participant in participants],
        *[f'revenue_{participant}' for participant in participants],
        *[f'profit_{participant}' for participant in participants],
        'water_level_salto',
        #    'production_excedentes',
        'marginal_cost',
        'demand'
    ]
    # Initialize results dictionary
    results_dict: dict = {}

    events_queries, _ = load_events()

    for query_name, query in events_queries.items():
        try:
            query_frequency = run_df.query(
                query).shape[0] / run_df.shape[0]
            results_dict[f'{query_name}_frequency'] = query_frequency
            for variable in variables:
                results_dict[f'{query_name}_{variable}'] = run_df.query(query)[
                    variable].mean()
        except KeyError as key_error:
            logger.error(
                'Query %s not successful, with KeyError: %s', query_name, key_error)
            logger.debug('Keys in run_df: %s', run_df.keys())
            logger.debug('Variables expected: %s', variables)
            continue
        except pd.errors.UndefinedVariableError as variable_error:
            logger.error(
                'Query %s not successful, with UndefinedVariableError: %s', query_name, variable_error)
            logger.debug('Variables expected: %s', variables)
            logger.debug('Variables in run_df: %s', run_df.keys())
            continue
    return results_dict


def std_variables(run_df: pd.DataFrame,
                  variables: list[str]) -> dict:
    """
    Compute the standard deviation of participant revenues with respect to scenarios.

    For each scenario, we first compute the mean revenue for each participant 
    (averaging over all time steps or data points in that scenario), and then 
    we compute the standard deviation of these scenario-level means across all scenarios.

    Arguments:
        run_df (pd.DataFrame): DataFrame containing at least a scenario identifier column 
                               and revenue columns for each participant.
        scenario_col (str): Name of the column that identifies scenarios in `run_df`.

    Returns:
        dict: A dictionary with keys as `participant_revenue_scenario_std` and values 
              as the standard deviation of scenario-level mean revenues.
    """
    # Ensure the scenario column exists
    if 'scenario' not in run_df.columns:
        raise ValueError("Column 'scenario' not found in run_df.")

    # Verify that all variables exist
    for variable in variables:
        if variable not in run_df.columns:
            raise ValueError(
                f"Missing variable column: '{variable}' in run_df.")

    # Group by scenario and compute the mean revenues for each participant in each scenario
    scenario_means = (run_df
                      .groupby('scenario')[variables]
                      .mean())

    # Compute the standard deviation of these scenario-level means across all scenarios
    results_dict = {}
    for variable in variables:
        scenario_std = scenario_means[variable].std()
        results_dict[f'{variable}_std'] = scenario_std

    return results_dict
