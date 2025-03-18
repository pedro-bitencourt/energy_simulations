import sys
sys.path.append('/projects/p32342/code')

from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging


setup_logging(level="info")

# Input parameters
name: str = 'battery_gas'
xml_basefile: str = '/projects/p32342/code/xml/battery_gas.xml'
costs_path: str = '/projects/p32342/code/cost_data/gas.json'

participants: list[str] = ['wind', 'solar', 'thermal', 'battery']

general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'cost_path': costs_path,
    'annual_interest_rate': 0.0,
    'email': 'pedro.bitencourt@u.northwestern.edu',
    'slurm': {'run': {'time': 1.2, 'mailtype': 'FAIL'}}
}

exog_grid: list[float] = [1, 2, 5, 10, 25, 50]

exogenous_variables: dict[str, dict] = {
    'battery_factor': {'grid': exog_grid},
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2200},
    'solar_capacity': {'initial_guess': 1500},
    'thermal_capacity': {'initial_guess': 1500}
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
# comparative_statics.prototype()
# comparative_statics.submit_solvers()
# Submit the processing job
comparative_statics.submit_processing()
#comparative_statics.redo_runs()
comparative_statics.process()
