
import numpy as np
from numpy.linalg import inv
from example_matrices import wind_revenue, solar_revenue, wind_capacity, solar_capacity

# Use the matrices and capacity arrays from the previous script
# wind_revenue, solar_revenue, wind_capacity, solar_capacity

def interpolate(x, x_values, y_values):
    i = np.searchsorted(x_values, x) - 1
    i = np.clip(i, 0, len(x_values) - 2)
    x0, x1 = x_values[i], x_values[i+1]
    y0, y1 = y_values[i], y_values[i+1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def revenue_function(wind_cap, solar_cap):
    wind_rev = interpolate(wind_cap, wind_capacity, wind_revenue[:, int(np.searchsorted(wind_capacity, wind_cap) - 1)])
    solar_rev = interpolate(solar_cap, solar_capacity, solar_revenue[int(np.searchsorted(solar_capacity, solar_cap) - 1), :])
    return np.array([wind_rev, solar_rev])

def jacobian(wind_cap, solar_cap, h=1.0):
    f_current = revenue_function(wind_cap, solar_cap)
    f_wind_plus_h = revenue_function(wind_cap + h, solar_cap)
    f_solar_plus_h = revenue_function(wind_cap, solar_cap + h)
    
    d_wind = (f_wind_plus_h - f_current) / h
    d_solar = (f_solar_plus_h - f_current) / h
    
    return np.column_stack((d_wind, d_solar))

def newton_method(target_wind, target_solar, initial_guess, max_iterations=100, tolerance=1e5):
    x = np.array(initial_guess)
    path = [x.copy()]
    
    print(f"Initial guess: Wind = {x[0]:.2f} MW, Solar = {x[1]:.2f} MW")
    for _ in range(max_iterations):
        f = revenue_function(x[0], x[1]) - np.array([target_wind, target_solar])
        print(f"Current error: {f}")
        if np.linalg.norm(f) < tolerance:
            break
        
        J = jacobian(x[0], x[1])
        print(f"Current Jacobian: {J}")
        dx = inv(J) @ f
        x = x - dx
        print(f"New unbounded guess: Wind = {x[0]:.2f} MW, Solar = {x[1]:.2f} MW")
        # find the nearest grid point
        def find_nearest(array, value):
            idx = np.abs(array - value).argmin()
            return array[idx]
        
        x[0] = find_nearest(wind_capacity, x[0])
        x[1] = find_nearest(solar_capacity, x[1])

        print(f"New guess: Wind = {x[0]:.2f} MW, Solar = {x[1]:.2f} MW")
        path.append(x.copy())
    
    return x, path

# Set target revenues and initial guess
target_wind = 2e6  # Example target wind revenue
target_solar = 2.2e6  # Example target solar revenue
initial_guess = [2000, 250]  # Initial guess for [wind_capacity, solar_capacity]

# Run Newton's method
result, path = newton_method(target_wind, target_solar, initial_guess)

print(f"Final result: Wind Capacity = {result[0]:.2f} MW, Solar Capacity = {result[1]:.2f} MW")
print("\nPath taken by the algorithm:")
for i, point in enumerate(path):
    print(f"Step {i}: Wind = {point[0]:.2f} MW, Solar = {point[1]:.2f} MW")
