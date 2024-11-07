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
