
import numpy as np
from investment_experiment_module import InvestmentExperiment
from constants import BASE_PATH

name: str = 'high_demand'
xml_basefile: str = f'{BASE_PATH}/code/xml/high_demand.xml'

run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                            'thermal': {'position': 1, 'multiplier': 1},
                            'wind': {'position': 2, 'multiplier': 1},
                            'solar': {'position': 3, 'multiplier': 1}}

general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': xml_basefile,
                            'name_function': run_name_function_params}

# create the grid of exogenous variables
current_hydro_capacity: int = 2215
current_thermal_capacity_per_module: int = 0

discrete_grid: list[float] = [1]
#discrete_grid: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
exogenous_variables: dict[str, dict] = {
    'hydro_factor': {'pattern': 'HYDRO_FACTOR'},
    'thermal': {'pattern': 'THERMAL_CAPACITY'}
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY'},
    'solar': {'pattern': 'SOLAR_CAPACITY'}
}

grid_hydro_factor: np.ndarray = np.array(discrete_grid)
grid_thermal_capacity: np.ndarray = (
    current_thermal_capacity_per_module + (1-grid_hydro_factor)*current_hydro_capacity/6)
# concatenate the two arrays
exogenous_variables_grid: dict[str, np.ndarray] = {'hydro_factor': grid_hydro_factor,
                                                   'thermal': grid_thermal_capacity}

# create the experiment
experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                  endogenous_variables, general_parameters)

results_df = experiment.recover_results()
print("Results so far:")
print(f'{results_df=}')

experiment.submit_jobs()
experiment.process_results()
