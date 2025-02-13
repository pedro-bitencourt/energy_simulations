from pathlib import Path

import src.finalization_module as fm
from src.utils.logging_config import setup_logging

setup_logging(level = "INFO")

simulations_folder: Path = Path(
    '/Users/pedrobitencourt/Projects/energy_simulations/sim/'
)

simulations: list[str] = [
    #    'qturbmax',
    #    'factor_compartir',
    'salto_volume'
]
    
x_variable = {
    'name': 'lake_factor', 'label': 'Lake Factor'
}

for simulation in simulations:
    results_path: Path = simulations_folder / simulation
    #    fm.finalize(results_path, x_variable)
#    fm.plot_results_summary(results_path, x_variable)
    fm.plot_optimal_capacities(results_path, x_variable)
    
