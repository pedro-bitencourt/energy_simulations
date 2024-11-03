"""
File name: run_experiment.py
Author: Pedro Bitencourt
Email: pedro.bitencourt@u.northwestern.edu
Description: This script submits multiple jobs using the MOP software.
"""
from pathlib import Path
import json
import shutil
import logging
import pandas as pd
from typing import Optional
import numpy as np

from auxiliary import submit_slurm_job, wait_for_jobs
from visualize_module import plot_stacked_price_distributions, plot_heatmap
from run_module import Run
from run_processor_module import RunProcessor
from experiment_visualizer_module import ExperimentVisualizer

from constants import (
    BASE_PATH)


def main():
    name: str = "mrs_experiment"

    run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                'thermal_capacity': {'position': 1, 'multiplier': 1}}

    # general parameters
    general_parameters: dict = {
        "xml_basefile": f"{BASE_PATH}/code/xml/mrs_experiment.xml",
        "name_function": run_name_function_params
    }

    # to change
    grid_hydro: np.ndarray = np.linspace(0.2, 1, 5)
    grid_thermal: np.ndarray = np.linspace(8, 100, 5)

    variables_grid: dict[str, np.ndarray] = {
        'hydro_factor': grid_hydro,
        'thermal_capacity': grid_thermal
    }
    variables: dict = {
        "hydro_factor": {"pattern": "HYDRO_FACTOR*"},
        "thermal_capacity": {"pattern": "THERMAL_CAPACITY"}
    }

    # initialize the experiment
    experiment = Experiment(
        name, variables, variables_grid, general_parameters)

    # run the experiment
    # experiment.submit_experiment()
    # experiment.process_experiment()

    # visualize the results
    #experiment.visualize_experiment()


class Experiment:
    """
    Represents an experiment. Creates multiple runs from a set of values and a
    .xml basefile.
    """

    def __init__(self,
                 name: str,
                 variables: dict[str, dict],
                 general_parameters: dict,
                 variables_grid: Optional[dict[str, np.ndarray]] = None,
                 runs_array: Optional[list[Run]] = None):
        self.name: str = name
        self.variables: dict[str, dict] = variables
        self.general_parameters: dict = general_parameters

        # initialize relevant paths
        self.paths: dict = self.initialize_paths()

        # create the input and output folders
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        if variables_grid:
            self.variables_grid: dict[str, np.ndarray] = variables_grid
            self.original_runs_array: list[Run] = None
        if runs_array:
            self.original_runs_array: list[Run] = runs_array

        # initialize the list of runs
        self.runs_array: list[Run] = self.initialize_runs()

    def process_experiment(self):
        # if experiment has more than 10 runs, submit jobs to cluster
        if len(self.runs_array) > 10:
            # for each run, submit a processor job
            job_ids = self.submit_processor_jobs()

            # wait for the jobs to finish
            wait_for_jobs(job_ids)
        else:
            for run in self.runs_array:
                run_processor = RunProcessor(run)
                results = run_processor.results_run()
                print(results)

        # gather the results
        results_df: pd.DataFrame = self.gather_results()

        # save the results to a .csv file
        results_df.to_csv(
            (f"{BASE_PATH}/result/{self.name}_results.csv"), index=False)

        results_df.to_csv(
            (f"{self.paths['output']}/{self.name}_results.csv"), index=False)
        # print the results
        print(f'{results_df=}')

    def submit_processor_jobs(self):
        job_ids: list = []
        for run in self.runs_array:
            run_processor = RunProcessor(run)
            if run_processor is None:
                logging.error("Run %s could not be processed.",
                              run.run_name)
                continue
            if run_processor.get_processed_status():
                logging.info("Run %s already processed", run.run_name)
                continue

            bash_script: Path = run_processor.create_bash_script()
            job_id = submit_slurm_job(bash_script)
            job_ids.append(job_id)
            logging.info("Submitted processor job with id %s for run %s",
                         job_id, run.run_name)
        return job_ids

    def gather_results(self):
        # initialize the results array
        results: list[dict] = []
        for run in self.runs_array:
            # load from the json results file
            json_path: Path = run.paths['results_json']

            if json_path.exists():
                with open(json_path, 'r') as file:
                    # append the results to the results array
                    results_run: dict = json.load(file)
                    if results_run is None:
                        print(f"Error: for {run.run_name} results_run is None")
                    results.append(results_run)
                # copy to experiment folder
                shutil.copy2(json_path, self.paths['output'])

            price_distribution_path: Path = run.paths['price_distribution']
            save_path: Path = self.paths['output'] / \
                f'{run.run_name}_price_distribution.csv'
            if price_distribution_path.exists():
                shutil.copy2(price_distribution_path, save_path)

        # transform results into a pandas dataframe
        results_df: pd.DataFrame = pd.DataFrame(results)

        return results_df

    def initialize_paths(self):
        paths = {}
        paths['input'] = Path(f"{BASE_PATH}/temp/{self.name}")
        paths['output'] = Path(f"{BASE_PATH}/output/{self.name}")
        paths['result'] = Path(f"{BASE_PATH}/result/{self.name}")
        return paths

    def initialize_runs(self):
        # if the runs array is already initialized, return it
        if self.original_runs_array:
            return self.original_runs_array

        # else, create the runs array from variables_grid
        # get length of the grids
        lengths: list = [len(grid) for grid in self.variables_grid.values()]

        # check if the grids have the same lengths
        if len(set(lengths)) != 1:
            logging.critical("Grid lengths: %s", lengths)
            logging.critical("Variables grid: %s", self.variables_grid)
            raise ValueError("The grids have different lengths. Aborting.")

        # get the length of the grid
        grid_length = lengths[0]

        # iterate over the grid
        runs_array: list[Run] = []
        for i in range(grid_length):
            variables = {
                key: {"pattern": self.variables[key]["pattern"],
                      "value": entry[i]}
                for key, entry in self.variables_grid.items()
            }
            # create a run
            runs_array.append(self.run_factory(variables))

        return runs_array

    def run_factory(self, variables: dict[str, dict]):
        return Run(self.paths['input'],
                   self.general_parameters,
                   variables,
                   experiment_folder=self.paths['output'])

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
        # extract parameters from the .xml file
        name_subfolder, initial_year, final_year = self.get_parameters()
        self.general_parameters.update({'name_subfolder': name_subfolder,
                                       'initial_year': initial_year,
                                        'final_year': final_year})

        # reinitialize the runs runs_array
        self.runs_array = self.initialize_runs()

        # submit all the runs
        successful = True
        for run in self.runs_array:
            job_id = run.submit_run()
            if job_id is None:
                successful = False

        return successful

    def visualize_experiment(self, grid_dimension: int):
        visualizer = ExperimentVisualizer(self.paths['output'])
        visualizer.visualize(grid_dimension)


if __name__ == "__main__":
    main()
