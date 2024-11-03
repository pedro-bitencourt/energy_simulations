import numpy as np
from investment_experiment_module import InvestmentExperiment
from constants import BASE_PATH

name: str = 'lake_capacity'
xml_basefile: str = f'{BASE_PATH}/code/xml/{name}.xml'

run_name_function_params = {'lake_factor': {'position': 0, 'multiplier': 10},
                            'thermal': {'position': 1, 'multiplier': 1},
                            'wind': {'position': 2, 'multiplier': 1},
                            'solar': {'position': 3, 'multiplier': 1}}

general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': xml_basefile,
                            'name_function': run_name_function_params}


# discrete_grid: list[float] = [0.6]
discrete_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {'pattern': 'LAKE_FACTOR', 'poly': True},
}

endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 1000}
}

grid_hydro_factor: np.ndarray = np.array(discrete_grid)

# concatenate the two arrays
exogenous_variables_grid: dict[str, np.ndarray] = {
    'lake_factor': grid_hydro_factor}

# create the experiment
experiment = InvestmentExperiment(name, exogenous_variables, exogenous_variables_grid,
                                  endogenous_variables, general_parameters)

results_df = experiment.recover_results()
print("Results so far:")
print(f'{results_df=}')

# experiment.submit_jobs()
experiment.process_results()
