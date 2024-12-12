import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import logging


logger = logging.getLogger(__name__)

RESULT_FILES = []
# This dictionary will store the events to be plotted, along with their labels
EVENTS: dict[str, str] ={ 
        'unconditional': 'Unconditional',
        'water_level_34': 'Salto water level < 34 m',
        'water_level_33': 'Salto water level < 33 m',
        'water_level_32': 'Salto water level < 32 m',
        'water_level_31': 'Salto water level < 31 m',
        'drought_25': 'Salto water level < 25 percentile',
        'drought_10': 'Salto water level < 10 percentile',
        'low_wind_25': 'Wind production < 25 percentile',
        'low_wind_10': 'Wind production < 10 percentile',
        'blackout_95': 'Lost load > 95 percentile',
        'blackout_99': 'Lost load > 99 percentile',
        'negative_lost_load': 'Lost load < -0.001',
        'blackout_positive': 'Lost load > 0.001',
        'profits_thermal_75': 'Thermal profits > 75 percentile',
        'profits_thermal_95': 'Thermal profits > 95 percentile',
    }

VARIABLES: list[str] = [
    'production_wind',
    'production_solar',
    'production_salto',
    'production_thermal',
    'production_total',
    'lost_load',
    'demand',
    'marginal_cost',
    'profits_thermal',
    'salto_capacity',
    'water_level_salto',
    'revenues_wind',
    'revenues_solar',
    'revenues_salto',
    'revenues_thermal',
]
COMPARISON_EVENTS: dict[str, list[dict]] = {
    'unconditional': [{'name': 'unconditional', 'label': 'Unconditional'}],
    'water_level': [{'name': 'unconditional', 'label': 'Unconditional'},
        {'name': 'water_level_34', 'label': 'Salto water level < 34 m'},
        {'name': 'water_level_33', 'label': 'Salto water level < 33 m'},
        {'name': 'water_level_32', 'label': 'Salto water level < 32 m'},
    ],
    'wind_production': [{'name': 'unconditional', 'label': 'Unconditional'},
        {'name': 'low_wind_25', 'label': 'Wind production < 25 percentile'},
        {'name': 'low_wind_10', 'label': 'Wind production < 10 percentile'},
    ],
    'lost_load': [{'name': 'unconditional', 'label': 'Unconditional'},
        {'name': 'blackout_95', 'label': 'Lost load > 95 percentile'},
        {'name': 'blackout_99', 'label': 'Lost load > 99 percentile'},
        {'name': 'blackout_positive', 'label': 'Lost load > 0.001'},
    ],
    'thermal_profits': [{'name': 'unconditional', 'label': 'Unconditional'},
        {'name': 'profits_thermal_75', 'label': 'Thermal profits > 75 percentile'},
        {'name': 'profits_thermal_95', 'label': 'Thermal profits > 95 percentile'},
    ],
}

VARIABLES_TO_PLOT: dict[str, list[dict]] = {
    'production': [{'name': 'production_wind', 'label': 'Average Production Wind (MW)'},
                   {'name': 'production_solar', 'label': 'Average Production Solar (MW)'},
                   {'name': 'production_salto', 'label': 'Average Production Hydro (MW)'},
                   {'name': 'production_thermal', 'label': 'Average Production Thermal (MW)'},
                   {'name': 'production_total', 'label': 'Average Production Total (MW)'},
                   {'name': 'lost_load', 'label': 'Average Lost Load (MW)'},
                   {'name': 'demand', 'label': 'Average Demand (MW)'},],
    'thermal_vs_hydro': [{'name': 'production_salto', 'label': 'Average Production Hydro (MW)'},
                         {'name': 'production_thermal', 'label': 'Average Production Thermal (MW)'}],
    'price': [{'name': 'marginal_cost', 'label': 'Average Price ($/MWh)'},],
    'profits': [{'name': 'profits_thermal', 'label': 'Average Variable Profits Thermal ($ per MW of capacity per hour)'},]
}

Y_AXIS_VARIABLE: dict = {
        'production': 'Production (MW)',
        'thermal_vs_hydro': 'Production (MW)',
        'price': 'Price ($/MWh)',
        'profits': 'Profits ($ per MW of capacity per hour)'
}



X_VARIABLE: dict = {
    'name': 'salto_capacity', 'label': 'Salto Capacity (MW)'
}

def visualize(results_folder: Path):
    logger.info("Starting the visualize() function... at path: " + str(results_folder))

    paths: dict[str, Path] = {
        'graphics': results_folder / 'graphics',
        'conditional_means': results_folder / 'conditional_means.csv',
    }

    # Check if results folder contains all the files
    for file in RESULT_FILES:
        if not (results_folder / file).exists():
            logger.error(f"File {file} not found in results folder.")

    # Load the conditional means dataframe
    conditional_means_df = pd.read_csv(paths['conditional_means'])

    # Create a dictionary with the data for each event
    events_data: dict = {
        event: conditional_means_df[[f'{event}_{var}' for var in VARIABLES]]
        for event in EVENTS.keys()
    }

    # Remove the prefixes
    for event, df in events_data.items():
        df.columns = [col.replace(f"{event}_", "") for col in df.columns]
        # Add the Salto capacity column
        df[X_VARIABLE['name']] = conditional_means_df[X_VARIABLE['name']]


    # Create line plots
    all_line_plots(events_data, paths['graphics'])

    # Format csv
    formatted_df = format_conditional_means(events_data)
    # Save to disk
    formatted_df.to_csv(paths['graphics'] / 'formatted_conditional_means.csv', index=False)
    
    pass

def all_line_plots(events_data: dict[str, pd.DataFrame], folder_path: Path):
    for comparison_name, set_events in COMPARISON_EVENTS.items():
        for plot_name, y_variables in VARIABLES_TO_PLOT.items():
            comparison_folder = folder_path / comparison_name
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"

            line_plot(events_data, set_events, y_variables, Y_AXIS_VARIABLE[plot_name] , X_VARIABLE, file_path)

def line_plot(events_data: dict[str, pd.DataFrame],
              events: list[dict],
              y_variables: list[dict],
              y_variable_axis: str,
              x_variable: dict,
              file_path: Path):
    """
    Create a line plot comparing different scenarios.

    Arguments:
    --- events_data: dict[str, pd.DataFrame] - A dictionary containing the data for each event.
    --- events: list[dict] - A list of dictionaries containing the name and label of each event.
    --- y_variables: list[dict] - A list of dictionaries containing the name and label of each variable.
    --- x_variable: dict - A dictionary containing the name and label of the x variable.
    
    """
    # Set up the plot style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create a color palette for the scenarios
    colors = sns.color_palette("husl", n_colors=len(events))
    
    # Plot each scenario
    for event, color in zip(events, colors):
        # Check if the event exists in the dataframe
        if event['name'] not in events_data.keys():
            logger.warning("Warning: %s not found in DataFrame", event)
            continue

        event_df = events_data[event['name']]

        # Sort the DataFrame by the x variable
        event_df = events_data[event['name']].sort_values(by=x_variable['name'])
        
        # Check if variables exist in DataFrame
        if x_variable['name'] not in event_df.columns:
            logger.warning("Warning: %s not found in DataFrame", x_variable)
            continue
        for y_variable in y_variables:
            if y_variable['name'] not in event_df.columns:
                logger.warning("Warning: %s not found in DataFrame", y_variable)
                continue
            
            plt.plot(event_df[x_variable['name']], 
                    event_df[y_variable['name']],
                    marker='o',
                    linestyle='-',
                    label=f"{event['label']}, {y_variable['label']}",
                    color=color)
    
            # Customize the plot
            plt.xlabel(x_variable['label'])
            plt.ylabel(y_variable_axis)
            plt.legend(title='Scenario/variable')
            plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plots
    plt.savefig(file_path, dpi=300)


def format_conditional_means(events_data: dict[str, pd.DataFrame]):
    def header(name):
        return pd.DataFrame([
            ['-' * 10 + f' {name} ' + '-' * 10],  # Header with dashes
            ['']  # Blank line
        ])
    # Prepare the output DataFrame
    final_df = pd.DataFrame()

    # Loop through each event
    for event in EVENTS.keys():
        # Add event header and append to final output
        event_header = header(EVENTS[event])
        final_df = pd.concat([final_df, event_header, events_data[event]], ignore_index=True)
        # Add a blank row as separator
        blank_row = pd.DataFrame({col: None for col in final_df.columns})
        final_df = pd.concat([final_df, blank_row], ignore_index=True)

    return final_df

