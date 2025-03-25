# Energy Simulations


## Patches
### 03.24.25
1. Completely refactored all the modules related to processing the raw files extracted from MOP, including: 
    - run_analysis_module.py
    - finalization_module.py
    - plotting_module.py
    - comparative_statics_module.py (excluded processing methods)
2. Changed the package structure, separating the MOP-wrapper package from the configurations, scripts, etc. 
3. Migrated the processing configurations to a single YAML file, `config.yaml`,
   which is read by the `mop_wrapper.src.load_configs` module.


## Introduction
This repository's code is a wrapper for the software Modelo Padron de Operacion (MOP), developed by UTE.
MOP takes an `.xml` file outlining the characteristics of an energy system as input, solves the optimal energy dispatch problem, and outputs the value function while simulating the energy system's operation using historical data from Uruguay.
The code in this repository allows the user to perform comparative statics exercises on the energy system characteristics, enabling the determination of plant capacities endogenously based on a zero-profit condition.

## Usage Guide

### Processing a finished exercise
After successfully running an exercise to convergence on Quest, you can process and graph the results 
using the `finalization_module` module and the `run_analysis_module` module.
All the plots are configured in the `config.yaml` file.

The steps for this are the following. First, you need to have completed the exercise and 
have extracted the raw data using the `ComparativeStatics.process()` method (see the next section for 
how to run an exercise). If you are running this script locally, you should be sure to have
    - All the raw data in the `sim/{exercise_name}/raw` folder
    - The solver results in the `sim/{exercise_name}/results` folder
Then, you can use the following script to process and visualize the results:

```python 
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path

from mop_wrapper.src.constants import BASE_PATH
from mop_wrapper.src.utils.logging_config import setup_logging
import mop_wrapper.src.finalization_module as fm

setup_logging('debug')

name: str = "factor_compartir_gas_hi_wind"
costs_path: Path = BASE_PATH / "code/cost_data/gas_high_wind.json"
x_variable: Dict[str, str] = {"name": "hydro_capacity", "label": "Hydro Capacity"}
participants: List[str] = ["solar", "wind", "thermal", "hydro"]

def pre_processing_function(run_data) -> Tuple[pd.DataFrame, Dict[str, float]]:
    run_df, capacities = run_data
    run_df.rename(columns={"production_salto": "production_hydro"}, inplace=True)
    capacities["hydro_capacity"] = 1620*capacities["factor_compartir"]
    return run_df, capacities

# Construct the SimulationData object
simulation_data: fm.SimulationData = fm.build_simulation_data(name, participants,
                                                        x_variable, costs_path,
                                                        pre_processing_function=pre_processing_function)

# Perform all the numerical analysis
results: pd.DataFrame = fm.default_analysis(simulation_data)
# Save results to disk
results.to_csv(BASE_PATH / f"sim/{name}/results.csv", index=False)
# Plot results
fm.plot_results(simulation_data, results)
# Plot densities of selected variables
fm.plot_densities(simulation_data)
```

This script will process the raw data from the exercise and generate a series of plots and tables with the results.
The main object is the `SimulationData` object, which contains all the necessary information for processing and plotting the results. 
To build an instance of it, you should use the `build_simulation_data` function, 
which takes arguments:

- `name` [str]: the name of the exercises
- `participants` [List[str]]: the list of participants in the exercise
- `x_variable` [Dict[str, str]]: a dictionary with the keys `name` and `label`, indicating the name of the exogenous variable and its label for plotting
- `costs_path` [Path]: the path to the cost data file
- `pre_processing_function` [Callable]: a function that takes the raw data and capacities and returns a processed DataFrame and capacities dictionary

### Running a New Exercise
To set up a comparative statics exercise, follow these steps:

1. Create an XML template file
2. Write a Python script to execute the program

First, you should decide the exact parameter in the `xml` configuration file to be your exogenous variable. Common choices include:

1. The capacity of power plants: modify the field `potMax` (units are MW)
2. The lake volume for hydropower plants

I've included a file `template.xml` that exhibits all the fields we have already used for our comparative statics exercises. The fields to be substituted are as ${expression}, where expression is any valid Python expression.

You should also define which variables will be endogenous and solved for using the zero-profit condition. Currently,
this is only possible with capacity.
After identifying the fields for comparative statics,
create an XML template and write a Python script to execute the program.

Let's examine a specific example: `expensive_blackout.py`,
which performs comparative statics over the volume of the lake of Salto Grande using a penalty of $20,000 for blackouts.

```python
"""
Comparative statics exercise for changing the volume of the lakes in the system.
"""
# Import modules from the src folder
import sys
sys.path.append('/projects/p32342/code')
from src.comparative_statics_module import ComparativeStatics
from src.utils.logging_config import setup_logging

# Set up logging
setup_logging(level="debug") # Other options: "info", "warning", "error", "critical"

# Input parameters
name: str = 'expensive_blackout'
xml_basefile: str = f'/projects/p32342/code/xml/{name}.xml'
cost_path: str = '/projects/p32342/data/costs_original.json'


general_parameters: dict = {
    'daily': True,
    'email': 'your.email@u.northwestern.edu', 
    'xml_basefile': xml_basefile,
    'cost_path': cost_path,
    'annual_interest_rate': 0.0,
    'slurm': None}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
exogenous_variables: dict[str, dict] = {
    'lake_factor': {
        'grid': exog_grid
    },
}

endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2000},
    'solar_capacity': {'initial_guess': 2000},
    'thermal_capacity': {'initial_guess': 1300}
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
comparative_statics.submit_solver()
# Extract the results
comparative_statics.process()
```

This script can then be run on the cluster by simply using the command `python expensive_blackout.py`.

The script requires several parameters:
- name [str]: name for the exercise, will be used for creating folders and files.
- `general_parameters` [dict]: a dictionary containing:
    - **email** (`str`): Email to receive notifications
    - **xml_basefile** (`str`): Path to the template XML file
    - **daily** (`bool`): Indicates if runs are daily (`True`) or weekly (`False`)
    - **annual_interest_rate** (`float`): Annual interest rate for investment problems
    - **slurm** (`dict`, optional): Dictionary containing options for SLURM
      - **run** (`dict`, optional): Options for run jobs
      - **solver** (`dict`, optional): Options for solver Jobs
      - **processing** (`dict`, optional): Options for processing job
         with each of them having keys
         - `mail-type` (`str`)
         - `time` (`float`)
         - `memory` (`int`)
    - **solver** (`dict`, optional): dictionary containing options for the solver
- variables [dict]: a dictionary containing:
    - **endogenous** [dict]: dictionary of endogenous variables. Entries are 
            variable name : dictionary with key `initial_guess`, which is the initial guess for the solver.
    - **exogenous** [dict]: dictionary of exogenous variables. Entries are 
            variable name : dictionary with key `grid`, which is the list of values for the exogenous variable.


### The zero-profit solver
The solver seeks to find roots of the profit functions:
$\pi_i(k) = R_i(k) - C_i(k)$
where:

- $i \in {\text{wind, solar, thermal}}$
- $R_i(k)$ is the average yearly revenue per MW of resource $i$ given capacities $k=(k_{\text{wind}}, k_{\text{solar}}, k_{\text{thermal}})$
- $C_i(k)$ is the average yearly cost per MW of resource $i$ given capacities $k$

Convergence is achieved when:
$|\frac{\pi_i(k)}{C_i(k)}| < 1%$
or 
$\pi_i(k)<0, k_i = 1$
That is, the ratio of average yearly profits to average yearly costs falls below 1%, or the yearly profit is negative and capacity is 1MW.


### Job Flow

When `comparative_statics.submit_solver()` executes, it submits jobs for each exogenous variable value to find zero-profit investment levels for endogenous variables. Jobs are named `f{name}_{value}`. For example, with lake_factor = 1, the job would be `expensive_blackout_problem_1`. Each solver job creates and submits `run` jobs for different endogenous variable combinations.

### Simulation Directory Structure
All the data generated by a simulation is stored in the `sim` folder. 
For example, `expensive_blackout` is stored in  `/sim/expensive_blackout/`.
The internal folder 

```
expensive_blackout/
├── raw/                          # Contains, for each solver, a .csv file containing all
                                  # variables extracted from MOP, including marginal cost,
                                  # production, demand, water level, etc. Is used by the
                                  # finalization_module to compute results and print graphs. 
│   ├── expensive_blackout_0.5.csv
│   ├── expensive_blackout_1.csv 
├── trajectories/                 # Folder for storing the trajectories of each solver.
│   ├── expensive_blackout_0.5_trajectory.json
│   ├── expensive_blackout_1_trajectory.json
├── expensive_blackout_0.5/
├── expensive_blackout_1/
│   ├── 1_1000_1000_300           # Run folder for specific parameters
│   │   ├── CAD-2024-DIARIA       # MOP's raw outputs
│   │   ├── 1_1000_1000_300.xml   # Run XML file
│   │   ├── 1_1000_1000_300.sh    # Run submission script
│   │   ├── 1_1000_1000_300.out   # Output log
│   │   ├── 1_1000_1000_300.err   # Error log
│   ├── 1_1010_1000_300
│   ├── 1_1000_1010_300
│   ├── 1_1000_1000_310
├── temp/                         # Contains: the bash script, .out and .err files for the exercise
```

## Documentation

### config Folder

Contains configuration files for the project in json format.

Files:
    - `config.yaml`: contains configurations, mostly for processing. 

### src Folder

Contains the main repository modules:

#### comparative_statics_module.py
The ComparativeStatics class takes a xml template, a list of endogenous variables,
a list of values for an exogenous variable, a cost data file, and other parameters, 
such as SLURM configurations.
It mainly serves to create and manage a list of Solver objects, one for each 
value of the exogenous variable.
    
#### solver_module.py
The Solver class takes a xml template, exogenous and endogenous variables

### xml Folder

Contains XML templates used by `ComparativeStatics`. Add new template files here for new exercises, including patterns for variable substitution.

### old_exercises Folder

Contains reference scripts from previous comparative statics exercises.
