from pathlib import Path

import src.finalization_module as fm
from src.utils.logging_config import setup_logging

setup_logging(level = "INFO")

simulations_folder: Path = Path(
    '/Users/pedrobitencourt/Projects/energy_simulations/sim/'
)

salto_capacity_variable = {
    'name': 'salto_capacity', 'label': 'Salto Capacity'
}
lake_factor_variable = {
    'name': 'lake_factor', 'label': 'Lake Factor'
}
simulations: dict[str,dict] = {
#    'qturbmax': salto_capacity_variable,
#    'factor_compartir': salto_capacity_variable,
#    'salto_volume': lake_factor_variable,
    'salto_volume_new': lake_factor_variable
}
    

for simulation, x_variable in simulations.items():
    results_path: Path = simulations_folder / simulation
    fm.finalize(results_path, x_variable)
    #fm.plot_results_summary(results_path, x_variable)
    #    fm.plot_optimal_capacities(results_path, x_variable)
    #fm.finalize_r(results_path, x_variable)
    #fm.plot_results_summary(results_path, x_variable)
  
