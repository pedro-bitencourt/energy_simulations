from pathlib import Path

import src.finalization_module as fm
from src.utils.logging_config import setup_logging

setup_logging(level = "INFO")

simulations_folder: Path = Path(
    '/Users/pedrobitencourt/Projects/energy_simulations/simulations/'
)

simulations: list[str] = [
    'qturbmax',
    'factor_compartir',
]
    
x_variable = {
    'name': 'salto_capacity', 'label': 'Hydropower Capacity (MW)'
}

for simulation in simulations:
    results_path: Path = simulations_folder / simulation
    fm.finalize(results_path, x_variable)
 #   fm.plot_results_summary(results_path, x_variable)
    
