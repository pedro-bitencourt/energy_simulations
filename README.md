# Energy Simulations

This repository's code is a wrapper for the software Modelo Padron de Operacion, developed by UTE. 
MOP takes as an argument a .xml file which outlines characteristics of an energy system, solves the optimal energy dispatch problem. 
MOP then outputs the value function and simulates the energy system's operation using historical data from Uruguay.
The code in this repository allows the user to perform comparative statics exercises on characteristics of the energy system, allowing for the capacity of some plants to be determined endogenously as the solution of a zero-profit condition. 

## src

This folder contains the main files of the repository. It is divided into the following modules, in rough order of importance:

### `comparative_statics_module.py`

This is the main module of the repository.
It contains the `ComparativeStatics` class, which is used to perform comparative statics exercises on the energy system.

Inputs:
    - name: name of the comparative statics exercise; is used to name the folder where the results are stored.
    - variables: dictionary containing the exogenous and endogenous variables. 
        Keys:
            o exogenous: list of exogenous variables.
            o endogenous: list of endogenous variables.
        Each variable is a dict with the following keys:
            o name (str): name of the variable.
            o pattern (str): a pattern in the xml template file to be substituted by the value of the variable.
            o initial_guess (optional, float): initial guess for the endogenous variables.
    - variables_grid: dictionary containing the grids for the exogenous variables. 
        Keys (str): are the names of the exogenous variables.
        Values (float): list of values for the exogenous variable.
    - general_parameters: dictionary containing the general parameters, with keys: 
        o xml_basefile (str): path to the template xml file.
        o daily (bool): boolean indicating if the runs are daily (True) or weekly (False).
        o name_subfolder (str): name of the subfolder where the runs are stored.
        o annual_interest_rate (float): annual interest rate for the investment problems.
        o years_run (float): number of years to run the investment problems.
        o requested_time_run (float): requested time for each MOP run to the cluster, in hours.

Public methods:
    - `submit`: for each value of the exogenous variables, it creates a Run (in the case of no endogenous variables), 
        or an InvestmentProblem (in the case of at least 1 endogenous variable) object and submits it to the cluster.
    - `process`: processes the results of the Runs, using the RunProcessor class. Results are stored in the `results` subfolder.
    - `visualize`: visualizes the results of the experiment. If the grid of the exogenous variables has 1 dimension, 
        it plots some y vs x graphs. If the grid has 2 dimensions, it plots heatmaps. The specific configurations for 
        these figures can be altered in the `constants.py` file. 

### `investment_module.py`
Creates a class `InvestmentProblem` that is used to solve the investment problem, that is, a zero-profit condition system for the endogenous variables.

Inputs:
    - folder: Path to the folder where the investment problem is stored.
    - exogenous_variables: Dictionary containing the exogenous variables.
    - endogenous_variables: Dictionary containing the endogenous variables.
    - general_parameters: Dictionary containing general parameters for the run.
        o all the parameters from the Run class.
        o requested_time_run: Requested time for the run.

Public methods:
    - `solve_investment`: solves the investment problem.
    - `equilibrium_run``: returns the equilibrium run if the last iteration converged.
    - `submit`: submits the investment problem to Quest.

## xml
This folder contains the xml template files that are used by the ComparativeStatics class. Whenever creating a new comparative statics exercise,
a new xml template file should be created in this folder. The template file should contain the pattern to be substituted by the value of the variables.

## old_exercises
This folder contains the scripts of previous comparative statics exercises. 
