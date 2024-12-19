"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""

import sys
sys.path.append('/projects/p32342/code')
import logging
from src.utils.logging_config import setup_logging
from src.comparative_statics_module import ComparativeStatics
# Import modules from the src folder


setup_logging(level=logging.DEBUG)

# Input parameters
name: str = 'salto_capacity'
xml_basefile: str = f'/projects/p32342/code/xml/{name}.xml'

general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'annual_interest_rate': 0.0,
    'slurm': {
        'run': {
            'email': 'pedro.bitencourt@u.northwestern.edu',
            'mail-type': 'FAIL',
            'time': 0.5,
            'memory': 8
        },
        'solver': {
            'email': 'pedro.bitencourt@u.northwestern.edu',
            'mail-type': 'END,FAIL',
            'time': 16.5,
            'memory': 8
        }
    }
}

exogenous_variable_grid: list[float] = [
    0, 0.2, 0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variable_grid: list[float] = [
    0, 0.2, 0.6, 0.75, 1.25, 1.5, 2, 3]
exog_grid: list[float] = [val*1620 for val in exogenous_variable_grid]
exog_grid: list[float] = [0, 324, 972]
exogenous_variables: dict[str, dict] = {
    'salto_capacity': {
        'grid': exog_grid
    },
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 1000},
    'solar_capacity': {'initial_guess': 1000},
    'thermal_capacity': {'initial_guess': 300}
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


# Submit the solver jobs
#comparative_statics.prototype()
comparative_statics.submit_solvers()
# Submit the processing job
# comparative_statics.submit_processing()
