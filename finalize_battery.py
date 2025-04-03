from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from mop_wrapper.src.constants import BASE_PATH
from mop_wrapper.src.utils.logging_config import setup_logging
from mop_wrapper.src import finalization_module as fm

setup_logging('debug')



name: str = "battery_gas"
costs_path: Path = BASE_PATH / "code/cost_data/gas.json"
x_variable: Dict[str, str] = {"name": "battery_factor", "label": "Battery Units"}
participants: List[str] = ["solar", "wind", "thermal"]

def pre_processing_function(run_data) -> Tuple[pd.DataFrame, Dict[str, float]]:
    run_df, capacities = run_data
    return run_df, capacities

simulation_data: fm.SimulationData = fm.build_simulation_data(name, participants,
                                                        x_variable, costs_path,
                                                        pre_processing_function=pre_processing_function,
                                                        test=False)

results: pd.DataFrame = fm.default_analysis(simulation_data)
# Save results to disk
results.to_csv(BASE_PATH / f"sim/{name}/results.csv", index=False)
# Plot results
fm.plot_results(simulation_data, results)
# Plot densities of selected variables
fm.plot_densities(simulation_data)
