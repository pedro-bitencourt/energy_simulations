"""
This module implements functions designed to analyze the results from a ComparativeStatics 
exercise. Typically, they will take as arguments:
- run_df: a pandas dataframe containing all the data extracted 
from MOP, with rows being an hour x scenario.
    Columns are: datetime, scenario, production_{p}, variable_costs_{p}, 
    demand, water_level (if hydro is present), for p in participants 
- capacities: a dictionary containing data created by the solver, 
such as optimal capacities, and the values for the exogenous variable. 
- costs: a dictionary containing data on the cost parameters used in the exercise.
- participants: a list of participant names.

The main function of the module is full_run_analysis, which orchestrates the work flow 
to compute all the results. Currently, the results are organized in: 
    - variable means
    - variable conditional means
    - standard deviations (over scenarios)
    - derived results (computed using other results as inputs)
"""

from typing import Tuple, Dict, Any, List, Callable, Union
import pandas as pd
import logging
import math

from .utils.auxiliary import log_exceptions

logger = logging.getLogger(__name__)

HYDRO_VARIABLES: List[str] = [
        # Marginal source variables
        "hydro_marginal = (active_hydro & ~active_thermal) | (active_hydro & price_mc_thermal_4000)",
        "thermal_marginal = (~active_hydro & active_thermal) | (active_thermal & price_mc_thermal)",
        "renewables_marginal = (~active_hydro & ~active_thermal)", 
        # Water level variables
        "water_level_31 = (water_level_salto < 31)",
        "water_level_33 = (water_level_salto < 33)",
        ]



# Custom types
RunData = Tuple[pd.DataFrame, Dict[str, Any]]

def full_run_analysis(run_data: Tuple[pd.DataFrame, Dict[str, Any]],
         participants: List[str],
         costs: Dict[str, Any],
         events_queries: Dict[str, Dict[str, str]],
         additional_results_fun: Union[Callable,  None] = None) -> Dict[str, Any]:
    run_df, capacities = run_data

    logger.debug("run_df: %s", run_df.head())
    logger.debug("capacities: %s", capacities)


    variables_to_create: List[str] = variables_to_create_function(participants, costs, 
                                                                  capacities)
    updated_run_df: pd.DataFrame = create_variables(run_df, variables_to_create)

    results: Dict[str, Any] = capacities.copy()
    results.update(costs)

    # Compute means for each numeric or boolean variable
    means_dict: Dict[str, Any] = compute_variable_means(updated_run_df)
    results.update(means_dict)

    conditional_means: Dict[str, Any] = compute_conditional_means(updated_run_df, participants, 
                                                                  events_queries)
    results.update(conditional_means)
    
    std_over_scenarios: Dict[str, Any] = compute_std_over_scenarios(updated_run_df,
                                                                    capacities, participants)
    results.update(std_over_scenarios)

    derived_results: Dict[str, Any] = compute_derived_results(results, participants)
    results.update(derived_results)

    if additional_results_fun:
        additional_results: Dict[str, Any] = additional_results_fun(run_data, results)
        results.update(additional_results)
    return results


def compute_variable_means(run_df: pd.DataFrame):
    return {f"mean_{var}": safe_mean(run_df, var) for var in 
                    run_df.select_dtypes(include=['boolean', 'number']).columns}


def create_variable_from_expression(expression: str, df: pd.DataFrame):
    try:
        return df.eval(expression, inplace=True)
    except Exception as e:
        logger.error("Error creating variable from expression: %s. Error message: %s",
                     expression, e)
        return None


def variables_to_create_function(participants: List[str], costs: Dict[str, Any],
                                 capacities: Dict[str, Any]) -> List[str]:
    # Core variables
    core_variables: List[str] = [
        "production_total = " + " + ".join([f"production_{p}" for p in participants]),
        *[f"revenue_{p} = production_{p} * marginal_cost" for p in participants],
        *[f"variable_cost_{p} = production_{p} * {costs[f'marginal_cost_{p}']}" for p in participants],
        *[f"active_{p} = (production_{p} > 0.1)" for p in participants],
        *[f"profit_{p} = (revenue_{p} - variable_cost_{p}) / {capacities[f'{p}_capacity']}" for p in participants],
        "production_minus_demand = production_total - demand",
        "excess_production = production_minus_demand.clip(lower=0)",
        "lost_load = production_minus_demand.clip(lower=0)",
        "fraction_excess = lost_load / production_total",
    ]

    other_variables: List[str] = [
        # Price distribution variables
        "price_4000 = (marginal_cost >= 4000)",
        f"price_mc_thermal = (marginal_cost > {costs['marginal_cost_thermal']} - 1) & (marginal_cost < {costs['marginal_cost_thermal']} + 1)",
        f"price_mc_thermal_4000 = ((marginal_cost > {costs['marginal_cost_thermal']})  & (marginal_cost < 4000))",
    ]
    if "hydro" in participants:
        other_variables.extend(HYDRO_VARIABLES)

    # Derived variables
    derived_variables: List[str] = [
        *[f"profit_{p}_price_4000 = (profit_{p} * price_4000)" for p in participants],
        *[f"profit_{p}_price_mc_thermal_4000 = (profit_{p} * price_mc_thermal_4000)" for p in participants],
        *[f"excess_{p} = production_{p}*fraction_excess" for p in participants],
    ]
    return (core_variables + other_variables + derived_variables)


def create_variables(run_df: pd.DataFrame,
                     variables_to_create: List[str]) -> pd.DataFrame:
    new_run_df: pd.DataFrame = run_df.copy()
    for var in variables_to_create:
        create_variable_from_expression(var, new_run_df)
    return new_run_df


def compute_std_over_scenarios(run_df: pd.DataFrame, capacities: Dict[str, Any],
                               participants: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for participant in participants:
        profit_by_scenario = run_df.groupby("scenario")[f"profit_{participant}"].mean()
        profit_std = float(profit_by_scenario.std())
        results[f"std_profit_{participant}"] = profit_std / capacities[f"{participant}_capacity"]
    return results


def compute_derived_results(results: dict[str, float], 
                            participants: List[str]) -> dict[str, float]:
    derived_results: dict[str, float] = {}
    derived_results["mean_revenue_total"] = sum(
        results[f"mean_revenue_{participant}"] for participant in participants
        )
    derived_results["mean_production_total"] = sum(
        results[f"mean_production_{participant}"] for participant in participants
    )
    derived_results["mean_capture_rate"] = (
        derived_results["mean_revenue_total"] / derived_results["mean_production_total"]
    )
    for participant in participants:
        derived_results[f"mean_capture_rate_{participant}"] = (
            results[f"mean_revenue_{participant}"] / results[f"mean_production_{participant}"]
        )
        derived_results[f"mean_profit_{participant}_per_mw"] = (
            results[f"mean_profit_{participant}"] / results[f"{participant}_capacity"]
        )
        derived_results[f"total_cost_{participant}"] = (
            results[f"fixed_cost_{participant}"] * results[f"{participant}_capacity"]
            + results[f"mean_variable_cost_{participant}"]
        )

    derived_results["total_cost"] = sum(
        derived_results[f"total_cost_{participant}"] for participant in participants
    )
    derived_results["lcoe"] = (
        derived_results["total_cost"] / results["mean_demand"]
    )
    return derived_results


def compute_conditional_means(run_df: pd.DataFrame,
                              participants: List[str],
                              events_queries: Dict[str, Dict[str, str]]) -> dict:
    variables: List[str] = variables_conditional_means(participants)
    results_list: List[Dict] = [
           compute_event(run_df, event_dict, variables)
           for event, event_dict in events_queries.items()
    ]
    results = {k: d[k] for d in results_list for k in d}
    return results


def variables_conditional_means(participants: List[str]) -> List[str]:
    """
    This function returns the list of variables for which we want to compute the conditional means.
    """
    variables: List[str] = [
        *[f"production_{p}" for p in participants],
        *[f"profit_{p}" for p in participants],
        "lost_load",
        "marginal_cost",
    ]
    return variables

@log_exceptions
def safe_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df.query(query)


@log_exceptions
def safe_mean(df: pd.DataFrame, variable: str) -> float:
    mean = df[variable].mean()
    if not math.isfinite(mean):
        logger.error("Non-finite value for mean_%s: %s", variable, mean)
    return mean



def compute_event(run_df: pd.DataFrame,
                  event_dict: Dict[str, Any],
                  variables: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    query: str = event_dict['query']
    event_name: str = event_dict['name']
    event_df: pd.DataFrame = safe_query(run_df, query)
    event_frequency = event_df.shape[0] / run_df.shape[0]
    results[f'mean_frequency_{event_name}'] = event_frequency
    for variable in variables:
        results[f'mean_{variable}_{event_name}'] = safe_mean(event_df, variable)
        # Check if the result is a finite number
        if not math.isfinite(results[f'mean_{variable}_{event_name}']):
            logger.error("Non-finite value for mean_%s_%s", variable, event_name)
            logger.debug("Query: %s", query)
            logger.debug("Event DataFrame: %s", event_df.head())
    return results
