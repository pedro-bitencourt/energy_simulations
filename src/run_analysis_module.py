import logging
from pathlib import Path
from typing import Any
import numpy as np

import pandas as pd

from .utils.statistics_utils import plot_density, plot_nonparametric_regression

logger = logging.getLogger(__name__)

SUBSAMPLE_SIZE = None



participants = ["wind", "solar", "hydro", "thermal"]

DENSITY_PLOTS = [
    {
        "column": "marginal_cost",
    },
    {
        "column": "marginal_cost",
        "filename": "marginal_cost_less_4000",
        "condition": lambda df: df["marginal_cost"] < 4000,
    },
    {
        "column": "marginal_cost",
        "filename": "marginal_cost_positive_hydro",
        "condition": lambda df: df["hydro_marginal"] == 1,
    },
    {
        "column": "net_demand",
    },
    {
        "column": "water_level_salto",
        "x_from": 29,
    },
    {
        "column": "production_hydro"
    },
    {
        "column": "production_thermal",
        "x_to": 1500,
        "condition": lambda df: df["production_thermal"] > 1,
    }
]

VARIABLES_TO_CREATE: list[str] = [
    # Continuous variables
    "production_total = production_wind + production_solar + production_hydro + production_thermal",
    "demand_minus_production = demand - production_total",
    "net_demand = demand - production_solar - production_wind",
    "lost_load = demand_minus_production.clip(lower=0)",
    # Participant variables
    *[
        f"revenue_{participant} = production_{participant} * marginal_cost"
        for participant in participants
    ],
    *[
        f"{participant}_active = (production_{participant} > 0.1)"
        for participant in participants
    ],
    "variable_cost_thermal = production_thermal * @MC_THERMAL",
    # Booleans: dispatch order
    "hydro_marginal = ((hydro_active == 1) & (thermal_active == 0))",
    "thermal_marginal = ((thermal_active == 1) & (hydro_active == 0))",
    "renewables_marginal = ((thermal_active == 0) & (hydro_active == 0))",
    # Booleans: price
    "price_4000 = (marginal_cost >= 4000)",
    "positive_lost_load = (demand > (production_total + 0.1))",
    # Note @MC_THERMAL is still allowed for external variables
    "price_mc_thermal_4000 = ((marginal_cost > @MC_THERMAL) & (marginal_cost < 4000))",
    # Booleans: water level
    "water_level_31 = (water_level_salto < 31)",
    "water_level_33 = (water_level_salto < 33)",
]

MEAN_VARIABLES: list[str] = [
    "production_total", "marginal_cost", "demand",
    "net_demand", "lost_load", "excess_production",
    "price_4000", "price_mc_thermal_4000", "positive_lost_load", 
    *[f"production_{participant}" for participant in participants],
    *[f"excess_{participant}" for participant in participants],
    *[f"revenue_{participant}" for participant in participants],
    "variable_cost_thermal",
    "hydro_active", "thermal_active", "water_level_31", "water_level_33",
    "hydro_marginal", "thermal_marginal", "renewables_marginal",
    "price_4000", "price_mc_thermal_4000",
    "lost_load"
]

VARIABLES_TO_QUANTILE: list[str] = [
    "price_4000", "price_mc_thermal_4000"
]

def analyze_run(
    data: pd.DataFrame,
    run_name: str,
    output_folder: Path | str,
    exogenous_variable_value: float,
    run_capacities: dict[str, float],
    fixed_cost_dict: dict[str, float],
    marginal_cost_dict: dict[str, float],
    overwrite: bool = False) -> dict[str, float]:

    output_folder: Path = Path(output_folder)

    logger.info("Starting analyze_run for %s", run_name)
    output_folder.mkdir(parents=True, exist_ok=True)


    # Step 0. Subsample 
    if SUBSAMPLE_SIZE is not None:
        data = data.sample(min(SUBSAMPLE_SIZE, len(data)))

    # Step 1. Create variables
    logger.info("Creating variables...")
    data = create_variables(data, marginal_cost_dict)

    # Step 2. Plot densities and nonparametric regression
    logger.info("Plotting densities and nonparametric regression...")
    plot_densities(data.copy(), output_folder, overwrite)
    plot_nonparametric_regression(data.copy(),
                                  y_variable="positive_lost_load",
                                  x_variable="water_level_salto",
                                  output_folder=output_folder,
                                  overwrite=overwrite)

    logger.info("Computing results...")
    results = compute_results(data.copy(),
                              run_name,
                              exogenous_variable_value,
                              run_capacities,
                              fixed_cost_dict,
                              marginal_cost_dict)
    return results

def compute_results(
    data: pd.DataFrame,
    run_name: str,
    exogenous_variable_value: float,
    capacities: dict[str, float],
    fixed_cost_dict: dict[str, float],
    marginal_cost_dict: dict[str, float]) -> dict[str, float]:

    logger.info("Computing results for run %s", run_name)
    results: dict[str, Any] = {}
    results["name"] = run_name
    results["exogenous_variable"] = exogenous_variable_value

    capacities_dict = capacities.iloc[0].to_dict()
    results.update(capacities_dict)

    logger.info("Computing means...")
    results.update(variable_means(data.copy()))

    logger.info("Computing expressions...")
    results.update(compute_from_expressions(data.copy(),
                                            capacities_dict,
                                            marginal_cost_dict))

    logger.info("Computing distributions over scenarios...")
    results.update(compute_distributions_over_scenarios(data.copy()))

    logger.info("Computing derived results...")
    results.update(compute_derived_results(results, fixed_cost_dict))
    return results

def create_variables(data: pd.DataFrame, marginal_cost_dict: dict) -> pd.DataFrame:
    # Hard coded
    data.rename(columns={"production_salto": "production_hydro"}, inplace=True)
    logger.debug("Column names: %s", data.columns)

    for expr in VARIABLES_TO_CREATE:
        logger.debug("Creating variable: %s", expr)
        data.eval(expr, inplace=True, engine="python", local_dict={"MC_THERMAL": marginal_cost_dict["thermal"]})

    # Hard coded, were not working with eval
    data["excess_production"] = (-data["demand_minus_production"]).clip(lower=0)
    data["excess_fraction"] = (data["excess_production"] /
        data["production_total"].replace({0: np.nan}))
    for participant in participants:
        data[f"excess_{participant}"] = data["excess_fraction"] * data[f"production_{participant}"]
    return data

def variable_means(data: pd.DataFrame) -> dict[str, float]:
    results: dict[str, float] = {}
    for var in MEAN_VARIABLES:
        logger.debug("Computing mean for variable: %s", var)
        results[f"mean_{var}"] = data[var].mean()
    return results

def compute_from_expressions(
    data: pd.DataFrame,
    capacities: dict[str, float],
    marginal_cost_dict: dict[str, float]
) -> dict[str, float]:
    results = {}

    for participant in participants:
        results[f"mean_capture_rate_{participant}"] = (
            data[f"revenue_{participant}"].mean() / data[f"production_{participant}"].mean()
        )
        # Skip hydro/salto
        if participant in ("hydro", "salto"):
            continue

        revenue_by_scenario = data.groupby("scenario")[f"revenue_{participant}"].mean()
        revenue_std = float(revenue_by_scenario.std())  # Convert to Python float
        results[f"std_revenue_{participant}"] = revenue_std / capacities[f"{participant}_capacity"]

        mc_participant = marginal_cost_dict.get(participant, 0) or 0

        results[f"marginal_cost_{participant}"] = mc_participant

        profit_expr = (data["marginal_cost"] - mc_participant) * data[f"production_{participant}"]


        results[f"{participant}_profit"] = float(profit_expr.mean()) / capacities[f"{participant}_capacity"]

        results[f"{participant}_profit_positive_ll"] = (
            (profit_expr * data["positive_lost_load"]).mean()
            / capacities[f"{participant}_capacity"]
        )
        results[f"{participant}_profit_price_4000"] = (
            (profit_expr * data["price_4000"]).mean()
            / capacities[f"{participant}_capacity"]
        )
        results[f"{participant}_profit_price_mc_thermal_4000"] = (
            (profit_expr * data["price_mc_thermal_4000"]).mean()
            / capacities[f"{participant}_capacity"]
        )

        data[f"{participant}_profit"] = profit_expr
        profit_by_scenario = data.groupby("scenario")[f"{participant}_profit"].mean()
        profit_std = float(profit_by_scenario.std())
        results[f"std_{participant}_profit"] = profit_std / capacities[f"{participant}_capacity"]

    results["mean_capture_rate"] = (
        (data["demand"] * data["marginal_cost"]).mean() / data["demand"].mean()
    )
    results["mean_capture_rate_production_weighted"] = (
        (data["production_total"] * data["marginal_cost"]).mean() / data["production_total"].mean()
    )

    return results

def compute_distributions_over_scenarios(data: pd.DataFrame) -> dict[str, float]:
    results: dict[str, float] = {}
    # Distributions over scenarios
    quantiles = [0.25, 0.50, 0.75]
    scenario_stats = data.groupby("scenario").agg({
        var: "mean" for var in VARIABLES_TO_QUANTILE
        }).reset_index()
    
    for var in VARIABLES_TO_QUANTILE:
        for quantile in quantiles:
            results[f"q{int(quantile*100)}_{var}"] = scenario_stats[var].quantile(quantile)
    return results


def compute_derived_results(results: dict[str, float], fixed_cost_dict: dict[str, float]) -> dict[str, float]:
    derived_results: dict[str, float] = {}
    # LCOE
    total_cost = (
        results["thermal_capacity"] * fixed_cost_dict["thermal"] +
        results["wind_capacity"] * fixed_cost_dict["wind"] +
        results["solar_capacity"] * fixed_cost_dict["solar"] +
        results["mean_variable_cost_thermal"]
    )

    mean_demand = results["mean_demand"]
    lcoe = total_cost / mean_demand
    derived_results["lcoe"] = lcoe

    return derived_results

def plot_densities(data: pd.DataFrame, output_folder: Path, overwrite: bool = False) -> None:
    for task in DENSITY_PLOTS:
        condition = task.get("condition", None)
        if condition is not None:
            sample = data.loc[condition(data)].copy()
        else:
            sample = data.copy()


        col_name = task["column"]
        x_from = task.get("x_from", None)
        x_to = task.get("x_to", None)
        bw = task.get("bw", 1.0)

        series: pd.Series = sample[col_name]


        filename = task.get("filename", col_name)
        out_path = output_folder / f"{filename}.png"

        plot_density(
            series,
            col_name,
            out_path,
            x_from=x_from,
            x_to=x_to,
            bandwidth=bw,
            overwrite=overwrite
        )

