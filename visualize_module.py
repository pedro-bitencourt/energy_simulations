"""
Name: visualize_module.py
Description: This module contains functions to visualize the outputs from an
experiment.
"""
from typing import Dict, List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from participant_module import REVENUE_STATS


# Configurations
RESULTS_TO_HEATMAP = []
RESULTS_TO_HEATMAP.append({
    'result_key': 'optimization_time_h',
    'label': 'Optimization Time in hours',
    'subfolder': 'general'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'simulation_time_h',
    'label': 'Simulation Time in hours',
    'subfolder': 'general'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'total_time_h',
    'label': 'Total Time in hours',
    'subfolder': 'general'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'avg_price_wind',
    'label': 'Average Price of a MWh for Wind (USD)',
    'subfolder': 'price'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'avg_price_solar',
    'label': 'Average Price of a MWh for Solar (USD)',
    'subfolder': 'price'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'avg_price_undiscounted_wind',
    'label': 'Average Price of a MWh for Wind (USD, not discounted)',
    'subfolder': 'price'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'avg_price_undiscounted_solar',
    'label': 'Average Price of a MWh for Solar (USD, not discounted)',
    'subfolder': 'price'
})
RESULTS_TO_HEATMAP.append({
    'result_key': 'avg_hours_blackout',
    'label': 'Average Hours of Blackout (all years)',
    'subfolder': 'general'
})

PARTICIPANTS_TO_PLOT = ['wind', 'solar', 'thermal']

for participant in PARTICIPANTS_TO_PLOT:
    RESULTS_TO_HEATMAP.append({
        'result_key': f'lcoe_{participant}',
        'label': f'LCOE for {participant} (USD/MWh)',
        'subfolder': 'general'
    })
    RESULTS_TO_HEATMAP.append({
        'result_key': f'lcoe_undiscounted_{participant}',
        'label': f'LCOE for {participant} (USD/MWh, not discounted)',
        'subfolder': 'general'
    })
    for stat in REVENUE_STATS:
        RESULTS_TO_HEATMAP.append({
            'result_key':
            f'{stat["key_prefix"]}_revenue_undiscounted_{participant}_per_mw',
            'label':
            f'{stat["label_prefix"]} Revenue per MW for {participant} (USD, not discounted)',
            'subfolder': 'revenues'
        })
        RESULTS_TO_HEATMAP.append({
            'result_key':
            f'{stat["key_prefix"]}_revenue_{participant}_per_mw',
            'label':
            f'{stat["label_prefix"]} Revenue per MW for {participant} (USD)',
            'subfolder': 'revenues'
        })

###############################################################################
# PLOT AGAINST RENEWABLES
###############################################################################
PLOT_AGAINST_RENEWABLES = []

PLOT_AGAINST_RENEWABLES.append({
    'y_variables': [
        {'key': 'avg_price_wind', 'label': 'Wind'},
        {'key': 'avg_price_solar', 'label': 'Solar'},
        {'key': 'avg_price_thermal', 'label': 'Thermal'}
    ],
    'ylabel': 'Average Price per MWh (USD)',
    'title': 'Average Price per a MWh (USD)',
    'filename': 'avg_price_per_mwh'
})

PLOT_AGAINST_RENEWABLES.append({
    'y_variables': [
        {'key': 'avg_revenue_wind_per_mw', 'label': 'Wind'},
        {'key': 'avg_revenue_solar_per_mw', 'label': 'Solar'},
        {'key': 'avg_revenue_all_thermal_per_mw', 'label': 'Thermal'}
    ],
    'ylabel': 'Average Revenue per MW (USD)',
    'title': 'Average Revenue per MW (USD)',
    'filename': 'avg_revenue_per_mwh'
})

##################################################
# VISUALIZE
##################################################
FILTERS_EXPERIMENT = {}
FILTERS_EXPERIMENT['none'] = [
    {'label': '', 'key': '', 'function': lambda df: df}
]
FILTERS_EXPERIMENT['thermal_nothermal'] = [
    {'label': 'without extra thermal capacity',
        'key': 'without_thermal_',
        'function': lambda df: df[df['thermal'] == 0]},
    {'label': 'with 2250MW extra thermal capacity',
        'key': 'with_thermal_',
        'function': lambda df: df[df['thermal'] == 2250]}
]


def visualize(self):
    """
    Visualize the results of the experiment.
    """
    subfolders_to_create = ['heatmaps', 'plots']
    for subfolder in subfolders_to_create:
        self.paths[subfolder] = create_folder(self.paths['result'], subfolder,
                                              remove_existing=True)
    self.heatmaps()
    self.plot_against_renewables()


def heatmaps(self):
    """
    Create heatmaps for the results of the experiment, following the
    configuration in RESULTS_TO_HEATMAP.
    """
    subfolders_to_create = ['general', 'price', 'revenues']
    for subfolder in subfolders_to_create:
        self.paths[subfolder] = create_folder(self.paths['heatmaps'], subfolder,
                                              remove_existing=True)
    results_df = pd.read_csv(self.paths['general_results'], sep='\t')
    # Hard coding some stuff, change later
    axes_variables = {'x': {'key': 'wind_capacity',
                            'label': 'Wind Capacity (MW)'},
                      'y': {'key': 'solar_capacity',
                            'label': 'Solar Capacity (MW)'}
                      }
    for filter_key in self.filters:
        filter_temp = self.FILTERS_EXPERIMENT[filter_key]
        for subfilter in filter_temp:
            df_temp = subfilter['function'](results_df)
            for result in RESULTS_TO_HEATMAP:
                variables = axes_variables.copy()
                variables.update({'z': {'key': result['result_key'], 'label':
                                        result['label']}})
                save_path = f"{self.paths['result']}/heatmaps/{result['subfolder']}/{result['result_key']}{subfilter['key']}"
                title = f"{result['label']} {subfilter['label']}"
                visualize_module.plot_heatmap(
                    df_temp, variables, save_path, title)

def plot_against_renewables(self):
    results_df = pd.read_csv(self.paths['general_results'], sep='\t')

    renewables = {'wind': 'Wind', 'solar': 'Solar'}
    other_renewable_dict = {'wind': 'solar', 'solar': 'wind'}
    current_capacities = {'wind': 2000, 'solar': 250}
    for filter_key in self.filters:
        filter_temp = self.FILTERS_EXPERIMENT[filter_key]
        for renewable, renewable_label in renewables.items():
            # filter for the median values in the other renewables
            other_renewable = other_renewable_dict[renewable]
            other_renewable_sq = current_capacities[other_renewable]
            for subfilter in filter_temp:
                df_temp = subfilter['function'](results_df)
                df_temp = df_temp[
                    df_temp[
                        other_renewable + '_capacity'
                    ] == other_renewable_sq]

                for plots in PLOT_AGAINST_RENEWABLES:
                    save_path = f"""{self.paths['result']}/
                        {plots['filename']}_{subfilter['key']}_{renewable}
                                 """
                    title = f"""{plots['ylabel']}
                        {subfilter['label']} """
                    x_variable = {'key': renewable + '_capacity',
                                  'label': renewable_label + ' Capacity (MW)'}
                    y_variables = plots['y_variables']
                    y_label = plots['ylabel']
                    save_path = f'{self.paths["result"]}/plots/{plots["filename"]}_{subfilter["key"]}{renewable}'
                    visualize_module.simple_plot(df_temp, x_variable,
                                                     y_variables, y_label,
                                                     save_path, title)

# Functions

def plot_heatmap(dataframe: pd.DataFrame, variables: Dict[str, Dict[str, str]],
                 save_path: str, title: str = None) -> None:
    """
    This function creates a heatmap from a dataframe.
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
