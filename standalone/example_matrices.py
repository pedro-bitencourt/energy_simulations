import numpy as np

# Wind revenue matrix
wind_revenue = np.array([
    [6.40e6, 6.20e6, 5.37e6, 3.96e6, 3.02e6, 2.47e6, 2.08e6, 1.78e6, 1.55e6, 1.36e6],
    [6.40e6, 6.18e6, 5.28e6, 3.23e6, 2.95e6, 2.41e6, 2.03e6, 1.74e6, 1.51e6, 1.32e6],
    [6.40e6, 6.14e6, 5.19e6, 3.78e6, 2.89e6, 2.36e6, 1.98e6, 1.70e6, 1.47e6, 1.28e6],
    [6.39e6, 6.10e6, 5.08e6, 3.67e6, 2.83e6, 2.31e6, 1.94e6, 1.65e6, 1.42e6, 1.23e6],
    [6.38e6, 6.04e6, 4.97e6, 3.62e6, 2.78e6, 2.27e6, 1.90e6, 1.61e6, 1.38e6, 1.19e6],
    [6.35e6, 5.97e6, 4.86e6, 3.56e6, 2.75e6, 2.24e6, 1.86e6, 1.58e6, 1.34e6, 1.16e6],
    [6.22e6, 5.75e6, 4.62e6, 3.48e6, 2.72e6, 2.20e6, 1.81e6, 1.51e6, 1.28e6, 1.08e6]
])

# Solar revenue matrix
solar_revenue = np.array([
    [3.22e6, 3.18e6, 3.04e6, 2.77e6, 2.56e6, 2.40e6, 2.26e6, 2.12e6, 2.00e6, 1.88e6],
    [3.22e6, 3.17e6, 2.98e6, 2.76e6, 2.48e6, 2.32e6, 2.19e6, 2.06e6, 1.93e6, 1.81e6],
    [3.22e6, 3.15e6, 2.92e6, 2.61e6, 2.39e6, 2.24e6, 2.11e6, 1.98e6, 1.86e6, 1.74e6],
    [3.22e6, 3.11e6, 2.83e6, 2.50e6, 2.29e6, 2.15e6, 2.02e6, 1.90e6, 1.78e6, 1.66e6],
    [3.21e6, 3.06e6, 2.72e6, 2.37e6, 2.18e6, 2.04e6, 1.92e6, 1.81e6, 1.69e6, 1.58e6],
    [3.19e6, 2.99e6, 2.59e6, 2.23e6, 2.04e6, 1.93e6, 1.82e6, 1.71e6, 1.59e6, 1.49e6],
    [3.09e6, 2.80e6, 2.27e6, 1.91e6, 1.74e6, 1.66e6, 1.57e6, 1.48e6, 1.38e6, 1.28e6],
    [3.09e6, 2.80e6, 2.27e6, 1.91e6, 1.74e6, 1.66e6, 1.57e6, 1.48e6, 1.38e6, 1.28e6]
])

# Wind capacity values
wind_capacity = np.array([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000])

# Solar capacity values
solar_capacity = np.array([125, 250, 375, 500, 625, 750, 875, 1000])

print("Wind Revenue Matrix:")
print(wind_revenue)
print("\nSolar Revenue Matrix:")
print(solar_revenue)
print("\nWind Capacity Values:")
print(wind_capacity)
print("\nSolar Capacity Values:")
print(solar_capacity)
