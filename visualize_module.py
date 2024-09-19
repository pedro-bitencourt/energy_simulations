"""
Name: visualize_module.py
Description: This module contains functions to visualize the outputs from an
experiment.
"""
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Dict, List, Union, Tuple
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def main():
    pass





def create_principal_components_plot(pca: PCA, n_components: int = 4) -> Tuple[plt.Figure, List[plt.Axes]]:
    fig, axs = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
    for i in range(n_components):
        axs[i].plot(pca.components_[i])
        axs[i].set_title(f'Principal Component {i+1}')
        axs[i].set_xlabel('Hour of Week')
        axs[i].set_ylabel('Weight')
    fig.tight_layout()
    return fig, axs

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

# Assuming price_distributions is your DataFrame
# main(price_distributions)
####################
# Plotting functions

def plot_intraweek_price_distribution(ax,
                                      dataframe: pd.DataFrame,
                                      title=None):
    ax.plot(dataframe['hour_of_week_bin'], dataframe['price_avg'])

    # Add vertical lines and labels for each day
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day in enumerate(days):
        ax.axvline(x=i*24, color='red', linestyle='--', alpha=0.5)
        ax.text(i*24 + 12, ax.get_ylim()[1], day, ha='center', va='bottom')

    ax.set_ylabel('Avg Price')
    
    # Add title with adjusted position
    if title:
        ax.set_title(title, pad=20)
    

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(0, 167)


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


def simple_plot(dataframe: pd.DataFrame, x_variable: Dict[str, str],
                y_variables: List[Dict[str, str]], y_label: str,
                save_path: str, title: str = None) -> None:
    """
    This function creates a simple plot from a dataframe.
    """
    plt.figure(figsize=(12, 6))
    for y_variable in y_variables:
        plt.plot(dataframe[x_variable['key']], dataframe[y_variable['key']],
                 label=y_variable['label'])
    plt.xlabel(x_variable['label'])
    plt.ylabel(y_label)
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(save_path)

if __name__ == "__main__":
    main()
