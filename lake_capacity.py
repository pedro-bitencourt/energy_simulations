"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
import numpy as np
from src.comparative_statics_module import ComparativeStatics


name: str = 'lake_capacity'
general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': 1.5}
exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {'pattern': 'LAKE_FACTOR'},
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 1000}
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

comparative_statics.submit()
# comparative_statics.process()
# comparative_statics.visualize(grid_dimension=1)
