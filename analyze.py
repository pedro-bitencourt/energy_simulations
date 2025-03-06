from typing import Any
import pandas as pd
from pathlib import Path
from src.utils.logging_config import setup_logging

setup_logging("info")

from src.comparative_statics_module import load_costs
from src.run_analysis_module import analyze_run

base_folder: Path = Path("/Users/pedrobitencourt/Projects/energy_simulations/")

experiment_name: str = "factor_compartir_coal"
exogenous_values: list[float] = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1]
cost_file: Path = base_folder / "code" / "cost_data" / "coal.json"

participants: list[str] = ["hydro", "wind", "solar", "thermal"]

# Paths
sim_folder: Path = base_folder/ "sim" / experiment_name
raw_folder: Path = sim_folder / "raw"
figures_folder: Path = base_folder/ "figures" / experiment_name
solver_results_path: Path = sim_folder/ "results" / "solver_results.csv"

# Read data
solver_results: pd.DataFrame = pd.read_csv(solver_results_path)
costs_dict: dict[str, dict[str, float]] = load_costs(cost_file)
fixed_cost_dict: dict[str, float] = costs_dict["fixed_cost_dictionary"]


results: list[dict[str, Any]] = []
for exogenous_value in exogenous_values:
    print("Analyzing exogenous value ", exogenous_value)
    
    run_name: str = f"{experiment_name}_{exogenous_value}"
    input_path: Path = raw_folder / f"{run_name}.csv"

    run_capacities = solver_results.loc[solver_results["name"] == run_name, ]

    print(run_capacities)

    run_data: pd.DataFrame = pd.read_csv(input_path)
    results_run = analyze_run(
        data=run_data,
        run_name=run_name,
        exogenous_variable_value=exogenous_value,
        run_capacities=run_capacities,
        fixed_cost_dict=fixed_cost_dict,
        output_folder=figures_folder,
        overwrite=True
    )
    results.append(results_run)

results_df: pd.DataFrame = pd.DataFrame(results)
results_df.to_csv(sim_folder / "results" / "results.csv", index=False)

