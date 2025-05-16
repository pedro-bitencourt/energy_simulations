"""
Description: this file computes results and creates plots for the exercise
"factor_compartir_gas_hi_wind". The key module used for this is the finalization_module, 
which implements functions to transform the raw MOP data stored in the 'raw' folder 
into results and graphs that can be used for analysis of the exercise.
"""
# Imports
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

# Local imports
from mop_wrapper.src.constants import BASE_PATH
from mop_wrapper.src.utils.logging_config import setup_logging
import mop_wrapper.src.finalization_module as fm

# Sets up the logging level, other options are: info, warning, critical
setup_logging('debug')

# Exercise related parameters:
name: str = "legacy_thermals_0.5"  # name of the exercise
costs_path: Path = BASE_PATH / \
    "code/cost_data/gas_legacy_thermals.json"  # cost filepath
# x_variable: "name" should contain the key for the exogenous variable of the exercise,
# while "label" should contain the label for the x-axis in the plots
x_variable: Dict[str, str] = {
    "name": "thermal_legacy", "label": "Legacy Thermal Capacity"}
# participants should contain all the participants present in the exercise
participants: List[str] = ["solar", "wind", "thermal"]

# pre_processing_function is optional and can be used for renaming variables and
# creating new ones before the analysis is performed. In this example,
# we use it to construct the hydropower capacity using the sharing factor.
def pre_processing_function(run_data) -> Tuple[pd.DataFrame, Dict[str, float]]:
    run_df, capacities = run_data
    capacities["thermal_legacy_capacity"] = capacities["thermal_legacy"]
    return run_df, capacities


# This line gathers all the relevant data and stores it into a SimulationData
# object
simulation_data: fm.SimulationData = fm.build_simulation_data(name, participants,
                                                              x_variable, costs_path,
                                                              pre_processing_function=pre_processing_function,
                                                              test=False)
# simulation_data.print()
# Analysis
# This line computes results from the raw dataframes and the capacity variables
#results: pd.DataFrame = fm.default_analysis(simulation_data)
# Save results to disk
#results.to_csv(BASE_PATH / f"sim/{name}/results.csv", index=False)
# Plot results
#fm.plot_results(simulation_data, results)
# Plot densities of selected variables
fm.plot_densities(simulation_data)
