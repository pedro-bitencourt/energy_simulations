
import numpy as np
from investment_experiment_module import InvestmentExperiment
from constants import BASE_PATH

name: str = 'battery_experiment'
xml_basefile: str = f'{BASE_PATH}/code/xml/battery_experiment.xml'

run_name_function_params = {'battery': {'position': 0, 'multiplier': 1},
                            'thermal': {'position': 1, 'multiplier': 1},
                            'wind': {'position': 2, 'multiplier': 1},
                            'solar': {'position': 3, 'multiplier': 1}}

general_parameters: dict = {'daily': False,
                            'name_subfolder': 'PRUEBA',
                            'xml_basefile': xml_basefile,
                            'name_function': run_name_function_params}


# discrete_grid: list[float] = [1]
discrete_grid: list[float] = [1, 450, 900, 1350, 1800]
exogenous_variables: dict[str, dict] = {
    'battery': {'pattern': 'BATTERY_STORAGE'},
}

endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 1000}
}

grid_hydro_factor: np.ndarray = np.array(discrete_grid)

# concatenate the two arrays
exogenous_variables_grid: dict[str, np.ndarray] = {
    'battery': grid_hydro_factor}

# create the experiment
experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                  endogenous_variables, general_parameters)

# results_df = experiment.recover_results()
# print("Results so far:")
# print(f'{results_df=}')

# experiment.submit_jobs()
experiment.process_results()
