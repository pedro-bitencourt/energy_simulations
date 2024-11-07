import sys
sys.path.append('..')
import numpy as np 
from experiment_module import Experiment
from constants import BASE_PATH

name: str = "mrs_experiment"

run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                            'thermal_capacity': {'position': 1, 'multiplier': 1}}

# general parameters
general_parameters: dict = {
    "xml_basefile": f"{BASE_PATH}/code/xml/mrs_experiment.xml",
    "name_function": run_name_function_params
}

# to change
grid_hydro: np.ndarray = np.linspace(0.2, 1, 5)
grid_thermal: np.ndarray = np.linspace(8, 100, 5)
# print shapes of the grids

variables_grid: dict[str, np.ndarray] = {
        'hydro_factor': grid_hydro, 
        'thermal_capacity': grid_thermal
}
variables: dict = {
                "hydro_factor": {"pattern": "HYDRO_FACTOR*"},
                "thermal_capacity": {"pattern": "THERMAL_CAPACITY"}
            }


# initialize the experiment
experiment = Experiment(name, variables, variables_grid, general_parameters)
#
## run the experiment
## experiment.submit_experiment()
## experiment.process_experiment()
#
## visualize the results
#experiment.visualize_experiment()
#
