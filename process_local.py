"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
BASE_PATH = '/Users/pedrobitencourt/Projects/energy_simulations/code'

import sys
import json

sys.path.append(BASE_PATH)
from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

setup_logging(level="debug")


# Input parameters
name: str = 'salto_volume'
xml_basefile: str = f'{BASE_PATH}/xml/{name}.xml'
costs_path: str = f'{BASE_PATH}/cost_data/original.json'

slurm_configs: dict = {
    'run': {
        'email': 'pedro.bitencourt@u.northwestern.edu',
        'mail-type': 'NONE',
        'time': 0.9,
        'memory': 3
    },
    'solver': {
        'email': 'pedro.bitencourt@u.northwestern.edu',
        'mail-type': 'END,FAIL',
        'time': 16.5,
        'memory': 8
    },
    'processing': {
        'email': 'pedro.bitencourt@u.northwestern.edu',
        'mail-type': 'ALL',
        'time': 5,
        'memory': 10
    }
}


general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'cost_path': costs_path,
    'annual_interest_rate': 0.0,
    'slurm': slurm_configs
}

exog_grid: list[float] = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2, 3, 5]
exog_grid: list[float] = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 2, 3, 5]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {'grid': exog_grid},
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2000},
    'solar_capacity': {'initial_guess': 1500},
    'thermal_capacity': {'initial_guess': 1200}
}

variables: dict[str, dict] = {
    'exogenous': exogenous_variables,
    'endogenous': endogenous_variables
}

# Create the ComparativeStatics object
comparative_statics = ComparativeStatics(
    name,
    variables,
    general_parameters
)


comparative_statics.compute_conditional_means()

