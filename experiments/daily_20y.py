# experiment_config.py
import numpy as np
import itertools
import sys
import experiment_module

name = 'static_20y'

daily = False

variables = ['wind_capacity', 'solar_capacity', 'thermal']

wind_current = 2000  # Get current wind capacity
solar_current = 250  # Get current solar capacity

grid = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
#grid = np.array([0, 1, 2])
range_wind = (wind_current * grid)
range_solar = (solar_current * grid)
range_thermal = [0, wind_current+solar_current]

set_of_values = [list(item) for item in itertools.product(range_wind, range_solar, range_thermal)]

pattern_variables = ['WIND_POTENCY', 'SOLAR_POTENCY', 'THERMAL_POTENCY']

aux_path = '/projects/p32342/code/xml/'
base_filename = '20y_static.xml'
base_file = rf'{aux_path}{base_filename}'

# Create an Experiment instance
experiment = experiment_module.Experiment(name, variables, set_of_values,
        pattern_variables, base_file, daily)
experiment.save()
