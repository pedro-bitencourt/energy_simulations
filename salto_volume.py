"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
import numpy as np
from src.comparative_statics_module import ComparativeStatics
from src.comparative_statics_visualizer_module import visualize
from src.constants import get_logger

from outer_constants import REQUESTED_TIME_SOLVER, REQUESTED_TIME_RUN

logger = get_logger(__name__)


name = 'salto_volume'
general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': 5.5,
                            'requested_time_solver': REQUESTED_TIME_SOLVER}

# exog_grid: list[float] = [0.75]
# exog_grid: list[float] = [0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 3, 5]
exog_grid: list[float] = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5]
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
visualize(comparative_statics, grid_dimension=1)
