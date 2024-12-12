import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
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
    'frequency',
    'production_wind',
    'production_solar',
    'production_salto',
    'production_thermal',
    'total_production',
    'lost_load',
    'demand',
    'marginal_cost',
    'profits_thermal',
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

VARIABLES_TO_PLOT: dict[str, dict] = {
    'production': {
        'title': 'Production (MW)',
        'variables': [
            {'name': 'production_wind', 'label': 'Wind'},
            {'name': 'production_solar', 'label': 'Solar'},
            {'name': 'production_salto', 'label': 'Hydro'},
            {'name': 'production_thermal', 'label': 'Thermal'},
            {'name': 'production_total', 'label': 'Total'},
            {'name': 'lost_load', 'label': 'Lost Load'},
            {'name': 'demand', 'label': 'Demand'},
        ]
    },
    'thermal_vs_hydro': {
        'title': 'Thermal vs Hydro Production (MW)',
        'variables': [
            {'name': 'production_salto', 'label': 'Hydro'},
            {'name': 'production_thermal', 'label': 'Thermal'}
        ]
    },
    'price': {
        'title': 'Price ($/MWh)',
        'variables': [
            {'name': 'marginal_cost', 'label': 'Price'}
        ]
    },
    'profits': {
        'title': 'Thermal Profits ($/MW-h)',
        'variables': [
            {'name': 'profits_thermal', 'label': 'Variable Profits'}
        ]
    }
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
        'investment_results': results_folder / 'investment_results.csv'
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
        for plot_name, plot_config in VARIABLES_TO_PLOT.items():
            comparison_folder = folder_path / comparison_name
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"
            title = f"{plot_config['title']} - {comparison_name}"

            line_plot(events_data, set_events, plot_config['variables'],
                      Y_AXIS_VARIABLE[plot_name] , X_VARIABLE, file_path,
                      title)
            
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
        event_data = events_data[event]
        # Order in salto_capacity
        event_data = event_data.sort_values(by=X_VARIABLE['name'])
        final_df = pd.concat([final_df, event_header, event_data], ignore_index=True)
        # Add a blank row as separator
        blank_row = pd.DataFrame({col: None for col in final_df.columns}, index=[0])
        final_df = pd.concat([final_df, blank_row], ignore_index=True)

    return final_df


def line_plot(events_data: dict[str, pd.DataFrame],
              events: list[dict],
              y_variables: list[dict],
              y_variable_axis: str,
              x_variable: dict,
              file_path: Path,
              title: str):
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
    
    # Create a colorblind-friendly color palette
    # Using a combination of IBM ColorBlind Safe palette and Color Brewer
    base_colors = [
        '#648FFF',  # Blue
        '#DC267F',  # Magenta
        '#785EF0',  # Purple
        '#FE6100',  # Orange
        '#FFB000',  # Gold
        '#009E73',  # Green
        '#56B4E9',  # Light Blue
        '#E69F00',  # Brown
        '#CC79A7',  # Pink
        '#000000'   # Black
    ]
    
    # Calculate total number of lines needed (events × variables)
    total_lines = len(events) * len(y_variables)
    
    # If we need more colors than in base_colors, create variations
    if total_lines > len(base_colors):
        # Create variations by adjusting lightness
        colors = []
        for color in base_colors:
            rgb = mcolors.to_rgb(color)
            hsv = mcolors.rgb_to_hsv(rgb)
            # Create a darker and lighter version of each color
            colors.extend([
                mcolors.hsv_to_rgb((hsv[0], hsv[1], min(1, hsv[2] * 0.7))),  # Darker
                rgb,
                mcolors.hsv_to_rgb((hsv[0], hsv[1], min(1, hsv[2] * 1.3)))   # Lighter
            ])
    else:
        colors = base_colors

    # Plot each scenario
    color_idx = 0
    for event in events:
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
                    color=colors[color_idx % len(colors)])
            
            color_idx += 1
    
    # Customize the plot
    plt.xlabel(x_variable['label'])
    plt.ylabel(y_variable_axis)
    plt.legend(title='Event/variable')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plots
    plt.savefig(file_path, dpi=300)
