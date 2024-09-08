# experiment_config.py
import numpy as np
import itertools
import sys
import experiment_module

name = 'hydros_to_thermal_weekly_control'
base_filename = 'hydros_to_thermal_weekly_control.xml'

daily = False 

variables_names = ['wind_capacity', 'solar_capacity']

wind_current = 2000  # Get current wind capacity
solar_current = 250  # Get current solar capacity

grid_wind = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2])
grid_solar = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4])

range_wind = grid_wind * wind_current
range_solar = grid_solar * solar_current

set_of_values = [
    list(item) for item in itertools.product(range_wind, range_solar)]

variables_patterns = ['WIND_POTENCY', 'SOLAR_POTENCY']

base_file = rf'/projects/p32342/code/xml/{base_filename}'

variables = {'names': variables_names,
        'patterns': variables_patterns}
filters = ['none']
# Create an Experiment instance
experiment = experiment_module.Experiment(name, variables, set_of_values,
                                         base_file, filters, daily)
experiment.save()
