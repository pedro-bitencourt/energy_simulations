from src.comparative_statics_module import ComparativeStatics
from src.comparative_statics_visualizer_module import visualize
import numpy as np
from rich.logging import RichHandler
import logging
import sys


# Configure the handler with pretty printing enabled
rich_handler = RichHandler(
    rich_tracebacks=True,
    show_time=True,
    show_path=True,
    markup=True
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    filename='salto_capacity.log',  # Name of the log file
    filemode='a',        # 'a' for append, 'w' for overwrite
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler]
)
# Ensure root logger uses the rich handler
logging.getLogger().handlers = [rich_handler]

# Option 2: If you want to suppress all matplotlib debugj
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
# Your module's logger
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Input parameters for the comparative statics exercise
name = 'salto_capacity'
general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': REQUESTED_TIME_RUN,
                            'requested_time_solver': REQUESTED_TIME_SOLVER}
exogenous_variable_name: str = 'hydro_factor'
exogenous_variable_pattern: str = 'HYDRO_FACTOR'
exogenous_variable_label: str = 'Hydro Factor'
exogenous_variable_grid: list[float] = [
    0, 0.2, 0.6, 0.75, 1, 1.25, 1.5, 2, 3]
# exogenous_variable_grid: list[float] = [1.25]

# No need to change from here on
exogenous_variables: dict[str, dict] = {
    exogenous_variable_name: {'pattern': exogenous_variable_pattern,
                              'label': exogenous_variable_label},
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
}
exogenous_variables_grid: dict[str, np.ndarray] = {
    exogenous_variable_name: np.array(exogenous_variable_grid)}
variables: dict[str, dict] = {'exogenous': exogenous_variables,
                              'endogenous': endogenous_variables}

# Create the ComparativeStatics object
comparative_statics = ComparativeStatics(name,
                                         variables,
                                         exogenous_variables_grid,
                                         general_parameters)

# All action happens here
# comparative_statics.submit()
# comparative_statics.redo_equilibrium_runs()
comparative_statics.process()
visualize(comparative_statics, grid_dimension=1, check_convergence=True)
