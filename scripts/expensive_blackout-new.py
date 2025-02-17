"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
# Import modules from the src folder
import sys
sys.path.append('/projects/p32342/code')
from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

# Set up logging
setup_logging(level="debug") # Other options: "info", "warning", "error", "critical"

# Input parameters
name: str = 'expensive_blackout'
xml_basefile: str = f'/projects/p32342/code/xml/{name}.xml'
cost_path: str = '/projects/p32342/data/costs_original.json'

# Set the slurm configurations. The settings below are the default,
# so if any field is not provided the program will use these values.
slurm_config: dict = {
    'run': {
        'mail-type': 'NONE',
        'time': 0.8,
        'memory': 5
    },
    'solver': {
        'mail-type': 'END,FAIL',
        'time': 12,
        'memory': 5
    },
    'processing': {
        'mail-type': 'END,FAIL',
        'time': 5,
        'memory': 10
    }
}

general_parameters: dict = {
    'daily': True,
    'email': 'aschwerz@u.northwestern.edu', 
    'xml_basefile': xml_basefile,
    'cost_path': cost_path,
    'annual_interest_rate': 0.0,
    'slurm': slurm_config}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {
        'grid': exog_grid
    },
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2000},
    'solar_capacity': {'initial_guess': 2000},
    'thermal_capacity': {'initial_guess': 1300}
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

# comparative_statics.prototype()

# Submit the solver jobs
comparative_statics.submit_solvers()
# Submit the processing job
comparative_statics.submit_processing()
