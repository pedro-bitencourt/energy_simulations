import sys
sys.path.append('/projects/p32342/code')

from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

setup_logging(level="debug")

# Input parameters
name: str = 'factor_compartir_gas_hi_wind'
xml_basefile: str = '/projects/p32342/code/xml/salto_capacity.xml'
costs_path: str = '/projects/p32342/code/cost_data/gas_high_wind.json'

general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'cost_path': costs_path,
    'annual_interest_rate': 0.0,
    'email': 'pedro.bitencourt@u.northwestern.edu'
}

exog_grid: list[float] = [0.001, 0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 3, 4]

exogenous_variables: dict[str, dict] = {
    'factor_compartir': {'grid': exog_grid},
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


# Submit the solver jobs
#comparative_statics.prototype()
#comparative_statics.submit_solvers()
# Submit the processing job
#comparative_statics.submit_processing()
#comparative_statics.process()
