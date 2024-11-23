# Energy Simulations

This repository's code is a wrapper for the software Modelo Padron de Operacion (MOP), developed by UTE.  MOP takes an `.xml` file outlining the characteristics of an energy system as input, solves the optimal energy dispatch problem, and outputs the value function while simulating the energy system's operation using historical data from Uruguay.  The code in this repository allows the user to perform comparative statics exercises on the energy system characteristics, enabling the determination of plant capacities endogenously based on a zero-profit condition.

The steps to use this package to perform comparative statics exercise are the following. First, one should decide the exact parameter in the `xml` configuration file to be our exogenous variable. In the past, we have used:

1. The capacity of power plants: for this, we can change the field `potMax` (units here are MW)
2. The lake volume for hydropower plants

The user should also define the variables that are endogenous and will be solved for using the zero-profit condition. So far, this is only possible with capacity. Having defined that and identified the field (or fields) over which we're doing comparative statics, one should create a template for the `xml` files (the example should illustrate how to set up a template). Having the template, the user should write a `.py` script to execute the program. 

Let's go over a specific example: `expensive_blackout.py`, which performs comparative statics over the volume of the lake of Salto Grande using a penalty of $20,000 for blackouts.


```python
import numpy as np
from src.comparative_statics_module import ComparativeStatics
from src.comparative_statics_visualizer_module import visualize

name = 'expensive_blackout'
general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': 6.5,
                            'requested_time_solver': 16.5}

exog_grid: list[float] = [0.6, 0.75, 1, 1.25, 1.5, 2, 3]
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

comparative_statics.submit()
comparative_statics.process()
visualize(comparative_statics, grid_dimension=1, check_convergence=True)
```
First, we need to define a name for the exercise; it's usually good practice to name the .py and the .xml file with this name as well. The `general_parameters` dictionary contains the following entries:
  - **xml_basefile** (`str`): Path to the template XML file.
  - **daily** (`bool`): Indicates if the runs are daily (`True`) or weekly (`False`).
  - **name_subfolder** (`str`): Name of the subfolder where runs are stored.
  - **annual_interest_rate** (`float`): Annual interest rate for the investment problems.
  - **years_run** (`float`): Number of years to run the investment problems.
  - **requested_time_run** (`float`): Requested time for each MOP run on the cluster, in hours.
  - **email** (`str`): the email that will receive notifications about the simulations.



## src

This folder contains the main files of the repository. It is divided into the following modules, in rough order of importance:

### `comparative_statics_module.py`

This is the main module of the repository.  
It contains the `ComparativeStatics` class, which is used to perform comparative statics exercises on the energy system.

#### Inputs:

- **name**: Name of the comparative statics exercise; used to name the folder where results are stored.
- **variables**: Dictionary containing the exogenous and endogenous variables.  
  **Keys**:
  - **exogenous**: List of exogenous variables.
  - **endogenous**: List of endogenous variables.
  
  Each variable is a dictionary with the following keys:
  - **name** (`str`): Name of the variable.
  - **pattern** (`str`): Pattern in the XML template file to be substituted by the value of the variable.
  - **initial_guess** (optional, `float`): Initial guess for the endogenous variables.

- **variables_grid**: Dictionary containing the grids for the exogenous variables.  
  **Keys** (`str`): Names of the exogenous variables.  
  **Values** (`float`): List of values for the exogenous variable.

- **general_parameters**: Dictionary containing general parameters with the following keys:
  - **xml_basefile** (`str`): Path to the template XML file.
  - **daily** (`bool`): Indicates if the runs are daily (`True`) or weekly (`False`).
  - **name_subfolder** (`str`): Name of the subfolder where runs are stored.
  - **annual_interest_rate** (`float`): Annual interest rate for the investment problems.
  - **years_run** (`float`): Number of years to run the investment problems.
  - **requested_time_run** (`float`): Requested time for each MOP run on the cluster, in hours.

#### Public Methods:

- **`submit`**: For each value of the exogenous variables, creates a `Run` (if no endogenous variables) or an `InvestmentProblem` (if at least one endogenous variable) and submits it to the cluster.
- **`process`**: Processes the results of the runs using the `RunProcessor` class, storing results in the `results` subfolder.
- **`visualize`**: Visualizes experiment results. If the exogenous variable grid has 1 dimension, it plots `y` vs. `x` graphs. If the grid has 2 dimensions, it plots heatmaps. Configurations for these figures can be altered in the `constants.py` file.

### `investment_module.py`

This module creates the `InvestmentProblem` class, used to solve the investment problem, which is a zero-profit condition system for the endogenous variables.

#### Inputs:

- **folder**: Path to the folder where the investment problem is stored.
- **exogenous_variables**: Dictionary containing the exogenous variables.
- **endogenous_variables**: Dictionary containing the endogenous variables.
- **general_parameters**: Dictionary containing general parameters for the run, including:
  - All parameters from the `Run` class.
  - **requested_time_run**: Requested time for the run.

#### Public Methods:

- **`solve_investment`**: Solves the investment problem.
- **`equilibrium_run`**: Returns the equilibrium run if the last iteration converged.
- **`submit`**: Submits the investment problem to Quest.

## xml

This folder contains the XML template files used by the `ComparativeStatics` class. When creating a new comparative statics exercise, add a new XML template file to this folder. The template should include patterns to be replaced by variable values.

## old_exercises

This folder contains scripts from previous comparative statics exercises.
