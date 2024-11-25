# Energy Simulations

## Introduction

This repository's code is a wrapper for the software Modelo Padron de Operacion (MOP), developed by UTE. MOP takes an `.xml` file outlining the characteristics of an energy system as input, solves the optimal energy dispatch problem, and outputs the value function while simulating the energy system's operation using historical data from Uruguay. The code in this repository allows the user to perform comparative statics exercises on the energy system characteristics, enabling the determination of plant capacities endogenously based on a zero-profit condition.

## Usage Guide

### Set Up

To set up a comparative statics exercise, follow these steps:

1. Create an XML template file
2. Write a Python script to execute the program

First, you should decide the exact parameter in the `xml` configuration file to be your exogenous variable. Common choices include:

1. The capacity of power plants: modify the field `potMax` (units are MW)
2. The lake volume for hydropower plants

You should also define which variables will be endogenous and solved for using the zero-profit condition. Currently, this is only possible with capacity. After identifying the fields for comparative statics, create an XML template and write a Python script to execute the program.

Let's examine a specific example: `expensive_blackout.py`, which performs comparative statics over the volume of the lake of Salto Grande using a penalty of $20,000 for blackouts.

```python
"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
# Set up logging
import logging
from src.logging_config import setup_logging

# Change the global logging level if needed; options are:
# logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
logger = setup_logging(level=logging.INFO)

# Import modules from the src folder
from src.comparative_statics_module import ComparativeStatics
from src.comparative_statics_visualizer_module import visualize

# Input parameters
name: str = 'expensive_blackout'
xml_basefile: str = f'/projects/p32342/code/xml/{name}.xml'

general_parameters: dict = {
    'daily': True,
    'name_subfolder': 'CAD-2024-DIARIA',
    'xml_basefile': xml_basefile,
    'email': 'your.email@domain.com',
    'annual_interest_rate': 0.0,
    'years_run': 6.61,
    'requested_time_run': 6.5,
    'requested_time_solver': 16.5
}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {
        'pattern': 'LAKE_FACTOR',
        'label': 'Lake Factor',
        'grid': exog_grid
    },
}
endogenous_variables: dict[str, dict] = {
    'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
    'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
    'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
}

variables: dict[str, dict] = {
    'exogenous': exogenous_variables,
    'endogenous': endogenous_variables
}

# Create the ComparativeStatics object
comparative_statics = ComparativeStatics(
    name,
    variables,
    general_parameters
)
# Submit the solver jobs
comparative_statics.submit()
# Process the results
comparative_statics.process()
# Visualize the results
visualize(comparative_statics, grid_dimension=1, check_convergence=True)
```

The script requires several parameters:

1. A name for the exercise (used to name files and folders)
2. The `general_parameters` dictionary containing:
   - **xml_basefile** (`str`): Path to the template XML file
   - **daily** (`bool`): Indicates if runs are daily (`True`) or weekly (`False`)
   - **name_subfolder** (`str`): Name of the subfolder where runs are stored
   - **annual_interest_rate** (`float`): Annual interest rate for investment problems
   - **years_run** (`float`): Number of years to run investment problems
   - **requested_time_run** (`float`): Requested time for each MOP run on the cluster, in hours
   - **requested_time_solver** (`float`): Requested time for the solver job on the cluster, in hours
   - **email** (`str`, optional): Email for simulation notifications

3. Variable definitions in the `variables` dictionary with two entries:
   - **endogenous**: Dictionary of endogenous variables
   - **exogenous**: Dictionary of exogenous variables
   
   Each variable entry contains:
   - `pattern`: Pattern in the XML to be substituted
   - `grid` (exogenous only): List of values
   - `initial_guess` (endogenous only): Initial guess for the solver
   - `label`: Used to construct graphs in the visualization stage

### The zero-profit solver
The solver seeks to find roots of the profit functions:
$\pi_i(k) = R_i(k) - C_i(k)$
where:

$i \in {\text{wind, solar, thermal}}$
$R_i(k)$ is the average yearly revenue per MW of resource $i$ given capacities $k=(k_{\text{wind}}, k_{\text{solar}}, k_{\text{thermal}})$
$C_i(k)$ is the average yearly cost per MW of resource $i$ given capacities $k$

Convergence is achieved when:
$|\frac{\pi_i(k)}{C_i(k)}| < 1%$
or 
$\pi_i(k)<0, k_i = 1$
That is, the ratio of average yearly profits to average yearly costs falls below 1%, or the yearly profit is negative and capacity is 1MW.


### Job Flow

When `comparative_statics.submit()` executes, it submits jobs for each exogenous variable value to find zero-profit investment levels for endogenous variables. Jobs are named `f{name}_investment_{value}`. For example, with lake_factor = 1, the job would be `expensive_blackout_investment_problem_1`. Each solver job creates and submits `run` jobs for different endogenous variable combinations.

### Folder Structure

Inside the main exercise folder (e.g., /p32342/comparative_statics/expensive_blackout/):

```
expensive_blackout/
├── expensive_blackout_investment_0.6/
├── expensive_blackout_investment_0.75/
├── expensive_blackout_investment_1/
│   ├── 1_1000_1000_300           # Run folder for specific parameters
│   │   ├── CAD-2024-DIARIA       # MOP's raw outputs
│   │   ├── 1_1000_1000_300.xml   # Run XML file
│   │   ├── 1_1000_1000_300.sh    # Run submission script
│   │   ├── 1_1000_1000_300.out   # Output log
│   │   ├── 1_1000_1000_300.err   # Error log
│   ├── 1_1010_1000_300
│   ├── 1_1000_1010_300
│   ├── 1_1000_1000_310
│   ├── expensive_blackout_investment_1_trajectory.json  # Solver trajectory
```

### Processing and Visualization

The `comparative_statics.process()` method processes results for each exogenous variable level using investment levels from the last iteration in the trajectory JSON file. Results are saved in the `/p32342/results` folder. The folder structure is:

```
expensive_blackout/
├── results_table.csv         # Table with the main results
├── marginal_cost.csv         # Raw marginal cost data
├── production_solar.csv      # Raw solar production data
├── other files
```

The `results_table.csv` file contains the main results of the exercise. Rows correspond to a specific value for the exogenous variable. Currently, the columns are:

- `wind`, `solar`, `thermal`: the zero-profit level of capacities for each resource, in MW
- `profit_wind`, `profit_solar`, `profit_thermal`: the yearly profits for each resource, as a fraction of the yearly costs (in dollars / MW / year)
- `convergence_reached`: whether convergence was reached for that value of the exogenous variable
- `production_hydros`, `produciton_thermals`, etc: the total production for each resource over the entire timeline, in GWh
- `price_avg`: the simple average of the spot price of energy, in dollars/MWh
- `price_weighted_avg`: the average spot price of energy, weighted by demand, in dollars/MWh
  
The other files contain the raw data from the simulations; that is, data at the hour-scenario level.


## Documentation

### src Folder

Contains the main repository modules:

#### comparative_statics_module.py

Main module containing the `ComparativeStatics` class for performing comparative statics exercises.

##### Inputs:
- **name**: Exercise name for folder naming
- **variables**: Dictionary of exogenous and endogenous variables
  - **exogenous**: List of exogenous variables
  - **endogenous**: List of endogenous variables
  
  Each variable contains:
  - **name** (`str`): Variable name
  - **pattern** (`str`): XML template pattern
  - **initial_guess** (`float`, optional): Initial value for endogenous variables

##### Public Methods:
- **submit()**: Creates and submits runs or investment problems
- **process()**: Processes results using RunProcessor
- **visualize()**: Generates result visualizations

#### investment_module.py

Contains `InvestmentProblem` class for solving zero-profit conditions.

##### Inputs:
- **folder**: Storage path
- **exogenous_variables**: Exogenous variable definitions
- **endogenous_variables**: Endogenous variable definitions
- **general_parameters**: Run parameters

##### Public Methods:
- **solve_investment()**: Solves investment problem
- **equilibrium_run()**: Returns converged solution
- **submit()**: Submits cluster job

### xml Folder

Contains XML templates used by `ComparativeStatics`. Add new template files here for new exercises, including patterns for variable substitution.

### old_exercises Folder

Contains reference scripts from previous comparative statics exercises.



