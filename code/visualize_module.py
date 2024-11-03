"""
Name: visualize_module.py
Description: This module contains functions to visualize the outputs from an
experiment.
"""
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def main():
    pass


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


if __name__ == "__main__":
    main()
