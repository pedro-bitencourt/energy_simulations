import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import visualize_module as vis

def main():
    experiment_folder: Path = Path('/Users/pedrobitencourt/quest/data/renewables/inv_zero_mc_thermal')
    experiment_visualizer = ExperimentVisualizer(experiment_folder)
    experiment_visualizer.visualize(grid_dimension=1)




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
        price_distribution_path: Path = self.experiment_folder / \
            f"{self.name}_price_distribution.csv"
        results_path: Path = self.experiment_folder / \
            f"{self.name}_results.json"

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

        self.dict_of_run_results: dict[str,
                                       RunResults] = self.initialize_run_results()
        self.save_paths: dict[str, Path] = {
            'heatmaps': self.experiment_folder / "heatmaps",
            'price_distributions': self.experiment_folder / "price_distributions",
            'one_d_plots': self.experiment_folder / "one_d_plots"
        }
        # create the directories if they don't exist
        for _, path in self.save_paths.items():
            path.mkdir(parents=True, exist_ok=True)
        # load results_df
        results_df_path: Path = self.experiment_folder / f"{self.name}_results.csv"
        self.results_df: pd.DataFrame = pd.read_csv(results_df_path)

    def visualize(self, grid_dimension: int):
        self.plot_pca_price_distributions()
        self.plot_intraweek_price_distributions()

        if grid_dimension == 2:
            self.plot_heatmaps()
        elif grid_dimension == 1:
            self.one_d_plots()
            

    def initialize_run_results(self):
        '''
        Reads the results of the experiment into RunResults objects. 
        '''
        # from the experiment folder, read {run.name}_results.csv and
        # {run.name}_price_distribution.csv and store them in a dict of RunResults
        dict_of_run_results: dict[str, RunResults] = {}
        for file in self.experiment_folder.iterdir():
            # check if file contains the experiment name
            if self.name in file.name:
                continue
            # get the filename
            filename = file.stem
            if filename.endswith("_results"):
                run_name = filename.replace("_results", "")
                run_results = RunResults(run_name, self.experiment_folder)
                dict_of_run_results.update({run_name: run_results})

        # sort the dict by the run name
        dict_of_run_results: dict[str, RunResults] = dict(
            sorted(dict_of_run_results.items()))

        return dict_of_run_results

    def plot_pca_price_distributions(self):
        price_distributions = pd.concat(
            [run_results.price_distribution_wide()
             for run_results in self.dict_of_run_results.values()],
            keys=self.dict_of_run_results.keys(),
            names=['run']
        )
        # Flatten the multi-index
        price_distributions = price_distributions.reset_index(level='run')
        # Separate the 'run' column and use the rest for PCA
        runs = price_distributions['run']
        # transform runs to a list of strings
        runs = [str(run) for run in runs]

        price_data = price_distributions.drop('run', axis=1)


        # Perform PCA
        pca, pca_result, scaler = perform_pca(price_data)

        # save explained variance ratio
        with open(self.save_paths['price_distributions'] / f"{self.name}_explained_variance_ratio.txt", 'w') as file:
            file.write("Explained variance ratio:\n")
            file.write(str(pca.explained_variance_ratio_))

        # Plot principal components
        pc_fig, _ = vis.create_principal_components_plot(pca)
        eigenvalue_fig, _ = vis.create_eigenvalue_decay_plot(pca)
        components_df = vis.output_principal_components_for_runs(
            pca_result, runs)

        # add hydro_factor and thermal_capacity to the components_df
        components_df['hydro_factor'] = [self.dict_of_run_results[run]
                                         .results_dict['hydro_factor']
                                         for run in runs]
        if 'thermal_capacity' in self.dict_of_run_results[runs[0]].results_dict:
            components_df['thermal_capacity'] = [self.dict_of_run_results[run]
                                                 .results_dict['thermal_capacity']
                                                 for run in runs]
        # get the correlation matrix
        correlation_df = components_df.corr()

        # Drop the cross-correlation of the principal components
        correlation_df = correlation_df.drop(columns=[col for col in correlation_df.columns if 'PC' in col])


        # Save plots and data
        pc_fig.savefig(self.save_paths['price_distributions'] /
                       f"{self.name}_principal_components.png", dpi=300)
        eigenvalue_fig.savefig(self.save_paths['price_distributions'] /
                               f"{self.name}_eigenvalue_decay.png", dpi=300)
        components_df.to_csv(self.save_paths['price_distributions'] /
                             f"{self.name}_principal_components.csv")
        correlation_df.to_csv(self.save_paths['price_distributions'] / 
                            f"{self.name}_correlation_matrix_principal_components.csv")

    def plot_intraweek_price_distributions(self):
        plots_list = []
        for run_results in self.dict_of_run_results.values():
            price_distribution = run_results.price_distribution
            results_dict = run_results.results_dict
            if 'thermal_capacity' in results_dict:
                title = f"Hydro factor: {results_dict['hydro_factor']:.1f}, Thermal capacity: {results_dict['thermal_capacity']:.1f}"
            else:
                title = f"Hydro factor: {results_dict['hydro_factor']:.1f}"
            plots_list.append({'data': price_distribution, 'title': title})

        save_path = self.save_paths['price_distributions'] / \
            f"{self.name}_price_distributions.png"
        vis.plot_stacked_price_distributions(plots_list, save_path)

    def plot_heatmaps(self):
        for hm_config in HEATMAP_CONFIGS:
            heatmap_config = hm_config
            title = f"{self.name} - {heatmap_config['title']}"
            save_path = self.save_paths['heatmaps'] / \
                heatmap_config['filename']
            vis.plot_heatmap(
                self.results_df, heatmap_config['variables'], save_path, title)

    def one_d_plots(self):
        for plot_config in ONE_D_PLOTS_CONFIGTS:
            x_key = plot_config['x_key']
            y_variables = plot_config['y_variables']
            axis_labels = plot_config['axis_labels']
            save_path = self.save_paths['one_d_plots'] / plot_config['filename']
            title = plot_config.get('title', None)
            print(self.results_df)
            vis.simple_plot(self.results_df, x_key,
                            y_variables, axis_labels,
                            save_path, title)


HEATMAP_CONFIGS = [
    {'variables': {'x': {'key': 'hydro_factor', 'label': 'Hydro Factor'},
                   'y': {'key': 'thermal_capacity', 'label': 'Thermal Capacity'},
                   'z': {'key': 'price_avg', 'label': 'Profit'}},
     'filename': 'price_avg_heatmap.png',
     'title': 'Price Heatmap'}
]

ONE_D_PLOTS_CONFIGTS = [
    # Optimal capacities of wind and solar
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'wind', 'label': 'Optimal Wind Capacity'},
            {'key': 'solar', 'label': 'Optimal Solar Capacity'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Capacity (MW)'},
        'title': 'Optimal Wind and Solar Capacities',
        'filename': 'wind_solar_capacities.png'
    },
    # Total production by resources
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'total_production_hydros', 'label': 'Hydro'},
            {'key': 'total_production_thermals', 'label': 'Thermal'},
            {'key': 'total_production_combined_cycle',
                'label': 'Combined Cycle'},
            {'key': 'total_production_wind', 'label': 'Wind'},
            {'key': 'total_production_solar', 'label': 'Solar'},
            {'key': 'total_production_import_export',
                'label': 'Import/Export'},
            {'key': 'total_production_demand', 'label': 'Demand'},
            {'key': 'total_production_blackout', 'label': 'Blackout'},

        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Total Production (GWh)'},
        'title': 'Total Production by Resources',
        'filename': 'total_production.png'
    },
    # New thermal production
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'total_production_hydros', 'label': 'Hydro'},
            {'key': 'new_thermal_production', 'label': 'New Thermal'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Total Production (GWh)'},
        'title': 'Production of Hydros and New Thermal',
        'filename': 'total_production_new_thermal.png'
    },
    # Average price
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'price_avg', 'label': 'Average Price'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Price ($/MWh)'},
        'title': 'Average Price',
        'filename': 'average_price.png'
    }
]


def perform_pca(price_distributions: pd.DataFrame) -> Tuple[PCA, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    price_distributions.columns = price_distributions.columns.astype(str)
    scaled_data: np.ndarray = scaler.fit_transform(price_distributions)
    pca = PCA()
    pca_result: np.ndarray = pca.fit_transform(scaled_data)
    return pca, pca_result, scaler


if __name__ == "__main__":
    main()
