import sys
sys.path.append('/projects/p32342/code/mop_wrapper')

from src.utils.logging_config import setup_logging
from src.comparative_statics_module import ComparativeStatics

setup_logging(level="info")

# Input parameters
name: str = 'legacy_thermals_0.25'
xml_basefile: str = '/projects/p32342/code/xml/legacy_thermals_0.25.xml'
costs_path: str = '/projects/p32342/code/cost_data/gas.json'

participants: list[str] = ['wind', 'solar', 'thermal']

general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'cost_path': costs_path,
    'annual_interest_rate': 0.0,
    'email': "arthurschwerzcahali2028@u.northwestern.edu",
    'slurm': {'run': {'time': 1.2, 'mailtype': 'FAIL'}},
    'wine_path': '/home/xvw8173/.wine',
}

exog_grid: list[float] = [1000, 2000, 2500, 3000]

exogenous_variables: dict[str, dict] = {
    'thermal_legacy': {'grid': exog_grid},
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2700},
    'solar_capacity': {'initial_guess': 2400},
    'thermal_capacity': {'initial_guess': 10}
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
# comparative_statics.submit_solvers()
# Submit the processing job
comparative_statics.submit_processing()
comparative_statics.process()
