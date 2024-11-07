import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.run_module import Run  # Import the updated Run class


class ComparativeStaticsVisualizer:
    '''
    Reads the results of a comparative statics exercise from a specified folder. The results must be
    in a folder with the following structure:
        - {experiment_folder}/
            - {run_name}_results.json
            - {run_name}_price_distribution.csv
    '''

    def __init__(self, experiment_folder: Path):
        self.experiment_folder: Path = experiment_folder
        self.name: str = experiment_folder.name

        self.dict_of_runs: dict[str, Run] = self._initialize_runs()
        self.save_paths: dict[str, Path] = {
            'heatmaps': self.experiment_folder / "heatmaps",
            'price_distributions': self.experiment_folder / "price_distributions",
            'one_d_plots': self.experiment_folder / "one_d_plots"
        }
        # Create the directories if they don't exist
        for _, path in self.save_paths.items():
            path.mkdir(parents=True, exist_ok=True)
        # Load results_df
        results_df_path: Path = self.experiment_folder / \
            f"{self.name}_results.csv"
        self.results_df: pd.DataFrame = pd.read_csv(results_df_path)

    def visualize(self, grid_dimension: int):
        self._plot_pca_price_distributions()
        self._plot_intraweek_price_distributions()

        if grid_dimension == 2:
            self._plot_heatmaps()
        elif grid_dimension == 1:
            self._one_d_plots()

    def _initialize_runs(self):
        '''
        Reads the results of the experiment into Run objects.
        '''
        # From the experiment folder, find subfolders that correspond to runs
        dict_of_runs: dict[str, Run] = {}
        for run_folder in self.experiment_folder.iterdir():
            if run_folder.is_dir():
                run_name = run_folder.name
                # Skip the main results folder
                if run_name == self.name:
                    continue
                # Create a Run object
                run = Run(
                    folder=run_folder,
                    general_parameters={},  # Assuming general_parameters are not needed here
                    variables={}  # Assuming variables are not needed here
                )
                # Check if results exist
                if run.load_results() is not None and run.load_price_distribution() is not None:
                    dict_of_runs[run_name] = run

        # Sort the dict by the run name
        dict_of_runs: dict[str, Run] = dict(
            sorted(dict_of_runs.items()))

        return dict_of_runs

    def _plot_pca_price_distributions(self):
        price_distributions_list = []
        run_names = []
        for run_name, run in self.dict_of_runs.items():
            wide_df = run.load_price_distribution()
            if wide_df is not None:
                price_distributions_list.append(wide_df)
                run_names.append(run_name)
        if not price_distributions_list:
            print("No price distributions available for PCA.")
            return

        price_distributions = pd.concat(
            price_distributions_list,
            keys=run_names,
            names=['run']
        )
        # Flatten the multi-index
        price_distributions = price_distributions.reset_index(level='run')
        # Separate the 'run' column and use the rest for PCA
        runs = price_distributions['run'].tolist()

        price_data = price_distributions.drop('run', axis=1)

        # Perform PCA
        pca, pca_result, scaler = perform_pca(price_data)

        # Save explained variance ratio
        with open(self.save_paths['price_distributions'] / f"{self.name}_explained_variance_ratio.txt", 'w') as file:
            file.write("Explained variance ratio:\n")
            file.write(str(pca.explained_variance_ratio_))

        # Plot principal components
        pc_fig, _ = create_principal_components_plot(pca)
        eigenvalue_fig, _ = create_eigenvalue_decay_plot(pca)
        components_df = output_principal_components_for_runs(
            pca_result, runs)

        # Add variables to the components_df if available
        if self.dict_of_runs[runs[0]].variables:
            for var_name in self.dict_of_runs[runs[0]].variables.keys():
                components_df[var_name] = [self.dict_of_runs[run]
                                           .variables.get(var_name, {}).get('value', None)
                                           for run in runs]

        # Get the correlation matrix
        correlation_df = components_df.corr()

        # Drop the cross-correlation of the principal components
        correlation_df = correlation_df.drop(
            columns=[col for col in correlation_df.columns if 'PC' in col])

        # Save plots and data
        pc_fig.savefig(self.save_paths['price_distributions'] /
                       f"{self.name}_principal_components.png", dpi=300)
        eigenvalue_fig.savefig(self.save_paths['price_distributions'] /
                               f"{self.name}_eigenvalue_decay.png", dpi=300)
        components_df.to_csv(self.save_paths['price_distributions'] /
                             f"{self.name}_principal_components.csv")
        correlation_df.to_csv(self.save_paths['price_distributions'] /
                              f"{self.name}_correlation_matrix_principal_components.csv")

    def _plot_intraweek_price_distributions(self):
        plots_list = []
        for run_name, run in self.dict_of_runs.items():
            price_distribution = run.load_price_distribution()
            if price_distribution is None:
                continue
            variables = run.variables
            # Create a title based on available variables
            title_parts = []
            for var_name, var_info in variables.items():
                title_parts.append(f"{var_name}: {var_info.get('value', '')}")
            title = ", ".join(title_parts)
            plots_list.append({'data': price_distribution, 'title': title})

        save_path = self.save_paths['price_distributions'] / \
            f"{self.name}_price_distributions.png"
        plot_stacked_price_distributions(plots_list, save_path)

    def _plot_heatmaps(self):
        for hm_config in HEATMAP_CONFIGS:
            heatmap_config = hm_config
            title = f"{self.name} - {heatmap_config['title']}"
            save_path = self.save_paths['heatmaps'] / \
                heatmap_config['filename']
            plot_heatmap(
                self.results_df, heatmap_config['variables'], save_path, title)

    def _one_d_plots(self):
        for plot_config in ONE_D_PLOTS_CONFIGS:
            x_key = plot_config['x_key']
            y_variables = plot_config['y_variables']
            axis_labels = plot_config['axis_labels']
            save_path = self.save_paths['one_d_plots'] / \
                plot_config['filename']
            title = plot_config.get('title', None)
            print(self.results_df)
            simple_plot(self.results_df, x_key,
                        y_variables, axis_labels,
                        save_path, title)


def perform_pca(price_distributions: pd.DataFrame) -> Tuple[PCA, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    price_distributions.columns = price_distributions.columns.astype(str)
    scaled_data: np.ndarray = scaler.fit_transform(price_distributions)
    pca = PCA()
    pca_result: np.ndarray = pca.fit_transform(scaled_data)
    return pca, pca_result, scaler


def create_eigenvalue_decay_plot(pca: PCA) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Eigenvalue Decay')
    return fig, ax


def output_principal_components_for_runs(pca_result: np.ndarray, original_index: pd.Index, n_components: int = 4) -> pd.DataFrame:
    components_df = pd.DataFrame(
        pca_result[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=original_index
    )

    components_df.to_csv('principal_components_per_run.csv')
    print("Principal components for each run saved to 'principal_components_per_run.csv'")
    return components_df


def create_principal_components_plot(pca: PCA, n_components: int = 3) -> Tuple[plt.Figure, List[plt.Axes]]:
    fig, axs = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
    for i in range(n_components):
        # Create a DataFrame with the principal component data
        pc_df = pd.DataFrame({
            'hour_of_week_bin': range(167),
            'price_avg': pca.components_[i]
        })
        # Use plot_intraweek_price_distribution for each component
        plot_intraweek_price_distribution(
            axs[i], pc_df, title=f'Principal Component {i+1}')
    fig.tight_layout()
    return fig, axs
# Assuming price_distributions is your DataFrame
# main(price_distributions)
####################
# Plotting functions


def plot_intraweek_price_distribution(ax, dataframe: pd.DataFrame, title=None):
    # Plot the price distribution
    ax.plot(dataframe['hour_of_week_bin'],
            dataframe['price_avg'], color='#1f77b4')

    # Add vertical lines and labels for each day
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        ax.axvline(x=i*24, color='#d62728', linestyle='--', alpha=0.5)
        ax.text(i*24 + 12, ax.get_ylim()
                [1], day, ha='center', va='bottom', fontweight='bold')

    # Highlight day and night times
    for i in range(7):
        # Day time (6 AM to 6 PM)
        ax.axvspan(i*24 + 6, i*24 + 18, facecolor='#ffff99', alpha=0.3)
        # Night time (6 PM to 6 AM)
        ax.axvspan(i*24 + 18, i*24 + 30, facecolor='#e6e6e6', alpha=0.3)

    # Highlight peak usage times (assuming 7-9 AM and 5-7 PM are peak times)
    for i in range(7):
        ax.axvspan(i*24 + 7, i*24 + 9, facecolor='#ff9999', alpha=0.3)
        ax.axvspan(i*24 + 17, i*24 + 19, facecolor='#ff9999', alpha=0.3)

    ax.set_ylabel('Weight', fontweight='bold')
    ax.set_xlabel('Hour of Week', fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 167)

    # Add a legend
    ax.fill_between([], [], color='#ffff99', alpha=0.3, label='Day Time')
    ax.fill_between([], [], color='#e6e6e6', alpha=0.3, label='Night Time')
    ax.fill_between([], [], color='#ff9999', alpha=0.3, label='Peak Usage')
    ax.legend(loc='upper right', frameon=False)

    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Adjust y-axis to start from 0
    # ax.set_ylim(bottom=0)


def plot_stacked_price_distributions(plots_list: list,
                                     save_path: Union[str, Path]) -> None:
    '''
    Plots a list of intraweek price distributions in a stacked manner.
    Arguments:
        --- plots_list: list - A list of dictionaries containing the data and title
        in the following format:
            [{'data': pd.DataFrame, 'title': str}, ...]
    '''
    n_plots = len(plots_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3*n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    for i, (plot, ax) in enumerate(zip(plots_list, axes)):
        title: str = plot['title']
        data: pd.DataFrame = plot['data']
        plot_intraweek_price_distribution(ax, data, title)

        # Only show x-label for the bottom plot
        if i == n_plots - 1:
            ax.set_xlabel('Hour of the Week')

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300)


def plot_heatmap(dataframe: pd.DataFrame, variables: Dict[str, Dict[str, str]],
                 save_path: Union[str, Path], title: str = None) -> None:
    """
    This function creates a heatmap from a dataframe.
    Arguments:
    --- dataframe: pd.DataFrame - The dataframe to be plotted.
    --- variables: Dict[str, Dict[str, str]] - A dictionary with the keys 'x', 'y',
        and 'z', each containing a dictionary with the keys 'key' and 'label'. The 
        'key' key contains the column name in the dataframe and the 'label' keys
        contains the label to be displayed in the plot.
    --- save_path: str - The path to save the plot.
    --- title: str - The title of the plot.
    """
    pivot_table = dataframe.pivot_table(values=variables['z']['key'],
                                        index=variables['y']['key'],
                                        columns=variables['x']['key'])
    plt.figure(figsize=(10, 7))
    if variables['z'].get('label', None):
        cbar_kws = {'label': variables['z']['label']}
    else:
        cbar_kws = {}

    def format_func(value, tick_number):
        return f'{value:.2e}'
    heatmap = sns.heatmap(pivot_table,
                          # Use scientific notation with 2 decimal places (3 sig figs)
                          fmt='.2e',
                          cmap='RdYlBu_r',
                          cbar_kws=cbar_kws,
                          annot=True,
                          annot_kws={'size': 7})
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticklabels([format_func(tick, 0) for tick in cbar.get_ticks()])

    plt.xlabel(variables['x']['label'])
    plt.ylabel(variables['y']['label'])
    plt.gca().invert_yaxis()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def simple_plot(dataframe: pd.DataFrame, x_key: str,
                y_variables: List[Dict[str, str]],
                axis_labels: Dict[str, str],
                save_path: str, title: Optional[str] = None) -> None:
    """
    This function creates a simple plot from a dataframe.
    """
    plt.figure(figsize=(12, 6))
    # iterate over the y variables to plot
    for y_variable in y_variables:
        plt.plot(dataframe[x_key],
                 dataframe[y_variable['key']],
                 label=y_variable.get('label', None)
                 )
    plt.xlabel(axis_labels['x'])
    plt.ylabel(axis_labels['y'])
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(save_path)
