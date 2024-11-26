"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
# Set up logging
from src.comparative_statics_visualizer_module import visualize
from src.comparative_statics_module import ComparativeStatics
import logging
from src.logging_config import setup_logging

# Change the global logging level if needed; options are:
# logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
setup_logging(level=logging.INFO)

# Import modules from the src folder

# Input parameters
name: str = 'expensive_blackout'
xml_basefile: str = f'/projects/p32342/code/xml/{name}.xml'

general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': xml_basefile,
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': 6.5,
                            'requested_time_solver': 16.5}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {'pattern': 'LAKE_FACTOR',
                    'label': 'Lake Factor',
                    'grid': exog_grid},
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
}

variables: dict[str, dict] = {'exogenous': exogenous_variables,
                              'endogenous': endogenous_variables}

# Create the ComparativeStatics object
comparative_statics = ComparativeStatics(name,
                                         variables,
                                         general_parameters)
# Submit the solver jobs
# comparative_statics.submit()
# Process the results
comparative_statics.process()
# Visualize the results
visualize(comparative_statics, grid_dimension=1, check_convergence=True)
