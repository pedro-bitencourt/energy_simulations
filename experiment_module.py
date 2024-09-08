"""
File name: run_experiment.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: This script submits multiple jobs using the MOP software.
"""
import logging
import subprocess
import pandas as pd
from pathlib import Path
import numpy as np
from auxiliary import create_folder, try_get_file
from run_module import Run

from constants import (
    BASE_PATH)

REQUESTED_TIME: str = '3:30:00'


def main():
    name: str = "mrs_experiment"

    run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                'thermal': {'position': 1, 'multiplier': 1}}

    # general parameters
    general_parameters: dict = {
        "xml_basefile": f"{BASE_PATH}/code/xml/mrs_experiment.xml",
        "name_function": run_name_function_params
    }

    # to change
    grid_hydro: np.array = np.linspace(0.2, 1, 5)
    grid_thermal: np.array = np.linspace(8, 100, 5)

    # construct the variables grid
    variables_grid = []
    for hydro in grid_hydro:
        for thermal in grid_thermal:
            variables_temp: dict = {
                "hydro_factor": {"pattern": "HYDRO_FACTOR*", "value": hydro},
                "thermal": {"pattern": "THERMAL_CAPACITY", "value": thermal}
            }
            variables_grid.append(variables_temp)

    # initialize the experiment
    experiment = Experiment(name, variables_grid, general_parameters)

    # run the experiment
    experiment.submit_experiment()


class Experiment:
    """
    Represents an experiment. Creates multiple runs from a set of values and a
    .xml basefile.
    """

    def __init__(self, name: str,
                 variables_grid: dict[str, dict],
                 general_parameters: dict):
        self.name: str = name
        self.variables_grid: dict[str, dict] = variables_grid
        self.general_parameters: dict = general_parameters

        # initialize relevant paths
        self.paths: dict = self.initialize_paths()

        # create the input and output folders
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # extract parameters from the .xml file
        name_subfolder, initial_year, final_year = self.get_parameters()
        self.general_parameters.update({'name_subfolder': name_subfolder,
                                       'initial_year': initial_year,
                                        'final_year': final_year})

        # initialize the list of runs
        self.runs_array: list[Run] = self.initialize_runs()

    def initialize_paths(self):
        paths = {}
        paths['input'] = Path(f"{BASE_PATH}/temp/{self.name}")
        paths['output'] = Path(f"{BASE_PATH}/output/{self.name}")
        return paths

    def initialize_runs(self):
        runs_array: list[Run] = []
        for variables in self.variables_grid:
            print('variables:')
            print(variables)
            runs_array.append(self.run_factory(variables))
        return runs_array

    def run_factory(self, variables: dict[str, dict]):
        input_folder: str = self.paths['input']
        general_parameters: dict = self.general_parameters
        return Run(input_folder, general_parameters, variables, experiment_folder=self.paths['output'])

    def get_parameters(self):
        """
        Extract name of the run, the initial year and the final year
        from the .xml file
        """
        xml_basefile_filepath = self.general_parameters['xml_basefile']
        with open(xml_basefile_filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            name = content.split("<nombre>")[1].split("</nombre>")[0].strip()
            initial_year = int(content.split(
                "<inicioCorrida>")[1].split(" ")[2])
            final_year = int(content.split("<finCorrida>")[1].split(" ")[2])
        return name, initial_year, final_year

    def submit_experiment(self):
        """
        Submit all the runs of the experiment to the cluster.
        """
        successful = True
        for run in self.runs_array:
            job_id = run.submit_run(REQUESTED_TIME)
            if job_id is None:
                successful = False
        return successful

    def process_experiment(self):
        pass


if __name__ == "__main__":
    main()
