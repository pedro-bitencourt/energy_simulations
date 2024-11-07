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
            o name: name of the variable.
            o pattern: a pattern in the xml template file to be substituted by the value of the variable.
            o initial_guess (optional): initial guess for the endogenous variables.
    - variables_grid: dictionary containing the grids for the exogenous variables. 
        Keys: are the names of the exogenous variables.
        Values: list of values for the exogenous variable.
    - general_parameters: dictionary containing the general parameters, with keys: 
        o xml_basefile: path to the template xml file.
        o daily: boolean indicating if the runs are daily (True) or weekly (False).
        o name_subfolder: name of the subfolder where the runs are stored.
        o annual_interest_rate: annual interest rate for the investment problems.
        o years_run: number of years to run the investment problems.
        o requested_time_run: requested time for each MOP run to the cluster.

Public methods:
    - `submit`: for each value of the exogenous variables, it creates a Run (in the case of no endogenous variables), 
        or an InvestmentProblem (in the case of at least 1 endogenous variable) object and submits it to the cluster.
    - `process`: processes the results of the Runs, using the RunProcessor class. Results are stored in the `results` subfolder.
    - `visualize`: visualizes the results of the experiment. If the grid of the exogenous variables has 1 dimension, 
        it plots some y vs x graphs. If the grid has 2 dimensions, it plots heatmaps. The specific configurations for 
        these figures can be altered in the `constants.py` file. 
