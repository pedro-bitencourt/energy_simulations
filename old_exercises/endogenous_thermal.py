
import numpy as np
from investment_experiment_module import InvestmentExperiment
from constants import BASE_PATH

name: str = 'endogenous_thermal'
xml_basefile: str = f'{BASE_PATH}/code/xml/endogenous_thermal.xml'

run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                            'thermal': {'position': 1, 'multiplier': 1},
                            'wind': {'position': 2, 'multiplier': 1},
                            'solar': {'position': 3, 'multiplier': 1}}

general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': xml_basefile,
                            'name_function': run_name_function_params}


# discrete_grid: list[float] = [1]
discrete_grid: list[float] = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
exogenous_variables: dict[str, dict] = {
    'hydro_factor': {'pattern': 'HYDRO_FACTOR'},
}

endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 1000}
}

grid_hydro_factor: np.ndarray = np.array(discrete_grid)

# concatenate the two arrays
exogenous_variables_grid: dict[str, np.ndarray] = {
    'hydro_factor': grid_hydro_factor}

# create the experiment
experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                  endogenous_variables, general_parameters)

# results_df = experiment.recover_results()
# print("Results so far:")
# print(f'{results_df=}')

experiment.submit_jobs()
# experiment.process_results()
