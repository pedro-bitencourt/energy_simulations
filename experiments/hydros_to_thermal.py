# experiment_config.py
import numpy as np
import itertools
import sys
import experiment_module

name = 'hydros_to_thermal'

daily = True

variables = ['wind_capacity', 'solar_capacity', 'hydros', 'thermal']

wind_current = 2000  # Get current wind capacity
solar_current = 250  # Get current solar capacity

thermal_extra = 2215

hydros_list_all = 'bonete,baygorria,palmar,salto'
hydros_list_none = ''

grid_wind = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
grid_solar = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4])

range_wind = grid_wind * wind_current
range_solar = grid_solar * solar_current

set_of_values_wind_solar = [
    list(item) for item in itertools.product(range_wind, range_solar)]

set_of_values = [
    (a, b, c, d)
    for a, b in itertools.product(range_wind, range_solar)
    for c, d in [(hydros_list_none, thermal_extra)]#(hydros_list_all, 0), 
]

pattern_variables = ['WIND_POTENCY', 'SOLAR_POTENCY', 'HYDROS_LIST',
                     'THERMAL_POTENCY']

aux_path = '/projects/p32342/code/xml/'
base_filename = 'hydros_to_thermal.xml'
base_file = rf'{aux_path}{base_filename}'

# Create an Experiment instance
experiment = experiment_module.Experiment(name, variables, set_of_values,
                                          pattern_variables, base_file, daily)
experiment.save()
