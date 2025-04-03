from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from mop_wrapper.src.constants import BASE_PATH
from mop_wrapper.src.utils.logging_config import setup_logging
import mop_wrapper.src.finalization_module as fm

setup_logging('debug')



name: str = "factor_compartir_gas_hi_wind"
costs_path: Path = BASE_PATH / "code/cost_data/gas_high_wind.json"
x_variable: Dict[str, str] = {"name": "hydro_capacity", "label": "Hydro Capacity"}
participants: List[str] = ["solar", "wind", "thermal", "hydro"]

def pre_processing_function(run_data) -> Tuple[pd.DataFrame, Dict[str, float]]:
    run_df, capacities = run_data
    run_df.rename(columns={"production_salto": "production_hydro"}, inplace=True)
    capacities["hydro_capacity"] = 1620*capacities["factor_compartir"]
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
