# Energy Simulations

This repository's code is a wrapper for the software Modelo Padron de Operacion (MOP), developed by UTE.  
MOP takes an `.xml` file outlining the characteristics of an energy system as input, solves the optimal energy dispatch problem, and outputs the value function while simulating the energy system's operation using historical data from Uruguay.  
The code in this repository allows the user to perform comparative statics exercises on the energy system characteristics, enabling the determination of plant capacities endogenously based on a zero-profit condition solution.

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
