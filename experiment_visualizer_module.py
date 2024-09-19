import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import visualize_module as vis


@dataclass
class RunResults:
    '''
    Represents the results of a single run of the experiment.
    This object reads results stored in:
        - {name}_price_distribution.csv
        - {name}_results.json
    '''
    name: str
    experiment_folder: Path
    results_exist: bool = False

    def __init__(self, name: str, experiment_folder: Path):
        self.name = name
        self.experiment_folder = experiment_folder

        # Construct the paths to the price distribution and results files
        price_distribution_path: Path = self.experiment_folder/f"{self.name}_price_distribution.csv"
        results_path: Path = self.experiment_folder / f"{self.name}_results.json"

        # check if the files exist
        self.results_exist: bool = price_distribution_path.exists() and results_path.exists()

        # Read the price distribution and results dataframes
        self.price_distribution: pd.DataFrame = self.read_price_distribution(
            price_distribution_path)
        self.results_dict: dict = self.read_results(results_path)

    def read_results(self, results_path: Path) -> dict:
        with open(results_path, 'r') as file:
            results = json.load(file)
        return results

    def read_price_distribution(self, price_distribution_path: Path) -> pd.DataFrame:
        price_distribution: pd.DataFrame = pd.read_csv(price_distribution_path)
        # the 'hour_of_week_bin' column is a string in the format "(start_hour, end_hour]"
        # convert it to just the start hour
        price_distribution['hour_of_week_bin'] = (
                price_distribution['hour_of_week_bin'].str.extract(r'(\d+)'))
        return price_distribution

    def price_distribution_wide(self) -> pd.DataFrame:
        # Pivot the table to have hours as columns
        wide_df = self.price_distribution.pivot_table(
            values='price_avg',
            index=None,
            columns='hour_of_week_bin'
        )
        # Convert all column names to integers and sort
        wide_df.columns = wide_df.columns.astype(int)
        wide_df = wide_df.sort_index(axis=1).reset_index(drop=True)
        return wide_df

class ExperimentVisualizer:
    '''
    Reads the results of an experiment from a specified folder. The results must be
    in a folder with the following structure, in order for correct parsing from the
    RunResults class:
        - {experiment_folder}/
            - {run_name}_results.json
            - {run_name}_price_distribution.csv
    '''
    def __init__(self, experiment_folder: Path):
        self.experiment_folder: Path = experiment_folder
        self.name: str = experiment_folder.name

        self.dict_of_run_results: dict[str, RunResults] = self.initialize_run_results()
        self.save_paths: dict[str, Path] = {
            'heatmaps': self.experiment_folder / "heatmaps",
            'price_distributions': self.experiment_folder / "price_distributions"
        }
        # create the directories if they don't exist
        for _, path in self.save_paths.items():
            path.mkdir(parents=True, exist_ok=True)
        # load results_df
        results_df_path: Path = self.experiment_folder / f"{self.name}.csv"
        self.results_df: pd.DataFrame = pd.read_csv(results_df_path)

    def initialize_run_results(self):
        '''
        Reads the results of the experiment into RunResults objects. 
        '''
        # from the experiment folder, read {run.name}_results.csv and
        # {run.name}_price_distribution.csv and store them in a dict of RunResults 
        dict_of_run_results: dict[str, RunResults] = {}
        for file in self.experiment_folder.iterdir():
            # get the filename
            filename = file.stem
            if filename.endswith("_results"):
                run_name = filename.replace("_results", "")
                run_results = RunResults(run_name, self.experiment_folder)
                dict_of_run_results.update({run_name: run_results})

        # sort the dict by the run name
        dict_of_run_results: dict[str, RunResults] = dict(sorted(dict_of_run_results.items()))

        return dict_of_run_results


    def plot_pca_price_distributions(self):
        price_distributions = pd.concat(
            [run_results.price_distribution_wide() for run_results in self.dict_of_run_results.values()],
            keys=self.dict_of_run_results.keys(),
            names=['run']
        )
        # Flatten the multi-index
        price_distributions = price_distributions.reset_index(level='run')
        # Separate the 'run' column and use the rest for PCA
        runs = price_distributions['run']
        price_data = price_distributions.drop('run', axis=1)
        # print the price_distribution dimensions
        print("Price distributions shape:")
        print(price_data.shape)
        # print the price_distribution head
        print("Price distributions head:")
        print(price_data.head())
        # get the principal components and eigenvalues for the price distributions
        pca, pca_result, scaler = perform_pca(price_distributions)
        print("Principal components:")
        print(pca.components_)
        print("Explained variance ratio:")
        print(pca.explained_variance_ratio_)
        print("Principal components shape:")
        print(pca_result.shape)
        # Perform PCA
        pca, pca_result, scaler = perform_pca(price_data)
        # Now proceed with plotting as before
        pc_fig, _ = vis.create_principal_components_plot(pca)
        eigenvalue_fig, _ = vis.create_eigenvalue_decay_plot(pca)
        components_df = vis.output_principal_components_for_runs(pca_result, runs)

        # add hydro_factor and thermal_capacity to the components_df
        components_df['hydro_factor'] = [self.dict_of_run_results[run]
            .results_dict['hydro_factor']
            for run in runs]
        components_df['thermal_capacity'] = [self.dict_of_run_results[run]
            .results_dict['thermal_capacity']
            for run in runs]
        # Save plots and data
        pc_fig.savefig(self.save_paths['price_distributions'] /
            f"{self.name}_principal_components.png", dpi=300)
        eigenvalue_fig.savefig(self.save_paths['price_distributions'] /
            f"{self.name}_eigenvalue_decay.png", dpi=300)
        components_df.to_csv(self.save_paths['price_distributions'] /
            f"{self.name}_principal_components.csv")

    def plot_intraweek_price_distributions(self):
        plots_list = []
        for run_results in self.dict_of_run_results.values():
            price_distribution = run_results.price_distribution
            results_dict = run_results.results_dict
            title = f"Hydro factor: {results_dict['hydro_factor']:.1f}, Thermal capacity: {results_dict['thermal_capacity']:.1f}"
            plots_list.append({'data': price_distribution, 'title': title})

        save_path = self.save_paths['price_distributions'] / f"{self.name}_price_distributions.png" 
        vis.plot_stacked_price_distributions(plots_list, save_path)

    def plot_heatmaps(self):
        for hm_config in HEATMAP_CONFIGS:
            heatmap_config = hm_config
            title = f"{self.name} - {heatmap_config['title']}"
            save_path = self.save_paths['heatmaps'] / heatmap_config['filename']
            vis.plot_heatmap(
                self.results_df, heatmap_config['variables'], save_path, title)


HEATMAP_CONFIGS = [
    {'variables': {'x': {'key': 'hydro_factor', 'label': 'Hydro Factor'},
                   'y': {'key': 'thermal_capacity', 'label': 'Thermal Capacity'},
                   'z': {'key': 'price_avg', 'label': 'Profit'}},
     'filename': 'price_avg_heatmap.png',
     'title': 'Price Heatmap'}
]
def perform_pca(price_distributions: pd.DataFrame) -> Tuple[PCA, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    price_distributions.columns = price_distributions.columns.astype(str)
    scaled_data: np.ndarray = scaler.fit_transform(price_distributions)
    pca = PCA()
    pca_result: np.ndarray = pca.fit_transform(scaled_data)
    return pca, pca_result, scaler
