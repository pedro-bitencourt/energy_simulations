import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.run_module import Run
from src.comparative_statics_module import ComparativeStatics
from src.constants import HEATMAP_CONFIGS, ONE_D_PLOTS_CONFIGS

logger = logging.getLogger(__name__)


def visualize(comparative_statics: ComparativeStatics, grid_dimension: int):
    visualizer = ComparativeStaticsVisualizer(comparative_statics)
    visualizer.visualize(grid_dimension)


class ComparativeStaticsVisualizer:
    '''
    '''
    def __init__(self, comparative_statics: ComparativeStatics):
        self.name: str = comparative_statics.name
        self.list_of_runs: list[Run] = comparative_statics.list_simulations

        self.exogenous_variables: dict = comparative_statics.exogenous_variables

        # Initialize the paths
        self.paths: dict[str, Path] = self._initialize_paths(
            comparative_statics.paths)

        # Create the directories if they don't exist
        for _, path in self.paths.items():
            path.mkdir(parents=True, exist_ok=True)

        # Load the results dataframe
        self.results_df: pd.DataFrame = pd.read_csv(
            self.paths['results'] / f"results_table.csv")

        self.results_runs: dict[str, dict] = self._initialize_results_runs()

    def _initialize_paths(self, paths: dict[str, Path]) -> dict[str, Path]:
        paths.update({
            'heatmaps': paths['results'] / "heatmaps",
            'price_distributions': paths['results'] / "price_distributions",
            'one_d_plots': paths['results'] / "one_d_plots"
        })
        return paths

    def _initialize_results_runs(self):
        """
        """
        results_runs: dict = {}
        for run in self.list_of_runs:
            # Initialize the results_run dictionary
            results_run: dict = {'price_distribution': run.load_price_distribution(),
                                 'variables': run.variables,
                                 }
            results_runs[run.name] = results_run
        return results_runs

    def visualize(self, grid_dimension: int):
        logger.info("Starting the visualize() function...")

        logger.info("Plotting PCA of the price distribution...")
        # Perform a PCA on the price distributions
        self._plot_pca_price_distributions()

        # Plot the intraweek price distributions
        self._plot_intraweek_price_distributions()

        if grid_dimension == 2:
            self._plot_heatmaps()
        elif grid_dimension == 1:
            self._one_d_plots()

    @staticmethod
    def get_pca(price_data: np.ndarray):
        """
        This function will perform PCA on the price data and return the principal components.
        No PCA object should be used outside of this function.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Do PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(price_data)
        
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        # Get the principal components
        pca_components: np.ndarray = pca.components_
        pca_explained_variance_ratio: np.ndarray = pca.explained_variance_ratio_
        return pca_components, pca_explained_variance_ratio, pca_result

    def _get_correlation_matrix(self, components_df: pd.DataFrame):
        """
        """
        correlation_df = components_df.corr()
        # Drop the cross-correlation of the principal components
        correlation_df = correlation_df.drop(
            columns=[col for col in correlation_df.columns if 'PC' in col]
        )
        return correlation_df

    def _get_components_df(self, pca_result: np.ndarray, runs: List[str], n_components: int):
        # First create DataFrame with just the PCA results
        components_df = pd.DataFrame(
            pca_result[:, :n_components],
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=runs  # Use the actual runs list as index
        )
        
        # Then add variables as columns using a dictionary comprehension
        variable_data = {
            var_name: [run.variables.get(var_name, {}).get('value', None) 
                      for run in self.list_of_runs]
            for var_name in self.list_of_runs[0].variables.keys()
        }
        
        # Update the DataFrame with the variable data
        components_df = components_df.assign(**variable_data)
        return components_df
    
    
    def _plot_pca_price_distributions(self):
        """
        Perform PCA on the price distributions, plot the principal components of each run,
        output the principal components to a CSV file
        """
        # Combine all price distributions
        price_distributions = pd.concat(
            [result_run['price_distribution'] 
             for result_run in self.results_runs.values()],
            keys=self.results_runs.keys(),
            names=['run']
        )
        # Get runs and data for PCA
        runs = price_distributions.index.get_level_values('run').unique().tolist()
        price_data: np.ndarray = price_distributions.values  
        pca_components, pca_explained_variance_ratio, pca_result = self.get_pca(price_data)

        # Save the explained variance ratio to a text file 
        with open(self.paths['price_distributions'] / f"{self.name}_explained_variance_ratio.txt", 'w') as file:
            file.write("Explained variance ratio:\n")
            file.write(str(pca_explained_variance_ratio))
    
        # Plot principal components
        pc_fig = create_principal_components_plot(pca_components)
        eigenvalue_fig = create_eigenvalue_decay_plot(pca_explained_variance_ratio)

        # Save the figures
        pc_fig.savefig(self.paths['price_distributions'] /
                       f"{self.name}_principal_components.png", dpi=300)
        eigenvalue_fig.savefig(self.paths['price_distributions'] /
                               f"{self.name}_eigenvalue_decay.png", dpi=300)

        ## Create a DataFrame with the principal component data
        #components_df = self._get_components_df(pca_result, runs, n_components=4)
        #correlation_df = self._get_correlation_matrix(components_df)
    
        ## Save the principal components and correlation matrix to CSV
        #components_df.to_csv(self.paths['price_distributions'] /
        #                     f"{self.name}_principal_components.csv")
        #correlation_df.to_csv(self.paths['price_distributions'] /
        #                      f"{self.name}_correlation_matrix_principal_components.csv")

    def _plot_intraweek_price_distributions(self):
        """
        Plots the average intraweek price distribution for each run. 

        The average is taken over both the scenarios and over the timeline.
        """
        plots_list = []
        for run_name, result_run in self.results_runs.items():
            # Create a title based on available variables
            title_parts = []
            for var_name, var_info in result_run['variables'].items():
                title_parts.append(f"{var_name}: {var_info.get('value', '')}")
            title = ", ".join(title_parts)
            plots_list.append(
                {'data': result_run['price_distribution'], 'title': title})

        save_path = self.paths['price_distributions'] / \
            f"{self.name}_price_distributions.png"
        plot_stacked_price_distributions(plots_list, save_path)

    def _plot_heatmaps(self):
        for hm_config in HEATMAP_CONFIGS:
            heatmap_config = hm_config
            title = f"{self.name} - {heatmap_config['title']}"
            save_path = self.paths['heatmaps'] / \
                heatmap_config['filename']
            plot_heatmap(
                self.results_df, heatmap_config['variables'], save_path, title)


    def _one_d_plots(self):
        """
        Plots results over a one-dimensional exogenous variable.
        """
        # Get the key for the exogenous variable
        x_key = [key for key in self.exogenous_variables.keys()][0]
        x_label = self.exogenous_variables[x_key]['label']
        # Iterate over plots; ONE_D_PLOTS_CONFIGS is in constants.py
        for plot_config in ONE_D_PLOTS_CONFIGS:
            # Unpack the plot configuration
            y_variables = plot_config['y_variables']
            axis_labels = plot_config['axis_labels']
            axis_labels['x'] = x_label
            save_path = self.paths['one_d_plots'] / \
                plot_config['filename']
            title = plot_config.get('title', None)
            # Create and save the plot
            simple_plot(self.results_df, x_key,
                        y_variables, axis_labels,
                        save_path, title)

################################################################################
# Plotting functions
################################################################################
def create_eigenvalue_decay_plot(pca_explained_variance_ratio: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.cumsum(pca_explained_variance_ratio))
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Eigenvalue Decay')
    return fig


def create_principal_components_plot(pca_components: np.ndarray, n_components: int = 3):
    fig, axs = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
    for i in range(n_components):
        logger.debug("Creating pc_df...")
        # Create a DataFrame with the principal component data
        print("{pca.components_[i]=}")
        pc_df = pd.DataFrame({
            'hour_of_week_bin': range(167),
            'price_avg': pca_components[i]
        })

        logger.debug("pc_df created")
        # Use plot_intraweek_price_distribution for each component
        plot_intraweek_price_distribution(
            axs[i], pc_df, title=f'Principal Component {i+1}')
    fig.tight_layout()
    return fig



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
