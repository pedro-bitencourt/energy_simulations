"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
import numpy as np
from src.comparative_statics_module import ComparativeStatics
from src.comparative_statics_visualizer_module import visualize
from rich.logging import RichHandler
import logging
import sys
from outer_constants import REQUESTED_TIME_RUN, REQUESTED_TIME_SOLVER

# Configure the handler with pretty printing enabled
rich_handler = RichHandler(
    rich_tracebacks=True,
    show_time=True,
    show_path=True,
    markup=True
)


# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler]
)
# Ensure root logger uses the rich handler
logging.getLogger().handlers = [rich_handler]

# Option 2: If you want to suppress all matplotlib debugj
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Your module's logger
logger = logging.getLogger(__name__)
name = 'expensive_blackout'
general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': REQUESTED_TIME_RUN,
                            'requested_time_solver': REQUESTED_TIME_SOLVER}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {'pattern': 'LAKE_FACTOR',
                    'label': 'Lake Factor'},
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
}
exogenous_variables_grid: dict[str, np.ndarray] = {
    'lake_factor': np.array(exog_grid)}
variables: dict[str, dict] = {'exogenous': exogenous_variables,
                              'endogenous': endogenous_variables}

# Create the ComparativeStatics object
comparative_statics = ComparativeStatics(name,
                                         variables,
                                         exogenous_variables_grid,
                                         general_parameters)

# comparative_statics.submit()
comparative_statics.process()
visualize(comparative_statics, grid_dimension=1, check_convergence=True)
