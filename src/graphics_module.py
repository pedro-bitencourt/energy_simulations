import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
import pandas as pd
import logging

from .utils.load_configs import load_events, load_plots, load_comparisons

logger = logging.getLogger(__name__)

# Load the events, variables, plots and comparisons
EVENTS: dict = load_events()
VARIABLES_TO_PLOT: dict = load_plots()
COMPARISON_EVENTS: dict = load_comparisons()

# TO FIX
VARIABLES = []


# This dictionary will store the events to be plotted, along with their labels
# Load events from events.json
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

    # Create relevant folders
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Visualize conditional means
    visualize_conditional_means(results_folder)

    # Plot optimal capacities
    plot_optimal_capacities(folder_path)
    pass

def visualize_conditional_means(folder_path: Path):
    # Plot event comparisons
    plot_event_comparisons(folder_path)

    # Format csv
    formatted_df = format_conditional_means(folder_path)

    # Save to disk
    formatted_df.to_csv(folder_path / 'formatted_conditional_means.csv', index=False)

def events_data_from_csv(folder_path: Path) -> dict[str, pd.DataFrame]:
    # Load the conditional means dataframe
    conditional_means_df = pd.read_csv(folder_path / 'conditional_means.csv')

    # Create a dictionary with the data for each event
    events_data: dict = {
        event: conditional_means_df[[f'{event}_{var}' for var in VARIABLES]]
        for event in EVENTS.keys()
    }

    # Remove the prefixes
    for event, df in events_data.items():
        df.columns = [col.replace(f"{event}_", "") for col in df.columns]

        # Get utilization rate for thermal and hydro
        df['utilization_thermal'] = df['production_thermal'] / df['thermal_capacity']
        df['utilization_salto'] = df['production_salto'] / df['salto_capacity']

    return events_data

def plot_event_comparisons(folder_path: Path):
    # Load events data from csv
    events_data = events_data_from_csv(folder_path / 'conditional_means.csv')

    for comparison_name, set_events in COMPARISON_EVENTS.items():
        for plot_name, plot_config in VARIABLES_TO_PLOT.items():
            comparison_folder = folder_path / comparison_name
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"
            title = f"{plot_config['title']} - {comparison_name}"

            line_plot(events_data, set_events, plot_config['variables'],
                      plot_config['y_label'], X_VARIABLE, file_path,
                      title)


def plot_optimal_capacities(folder_path: Path):
    # Load the investment results_folder
    investment_results = pd.read_csv(folder_path / 'investment_results.csv')
    events_data = {'none': investment_results}
    events = [{'name': 'none', 'label': ''}]
    y_variables = [{'name': 'wind_capacity_mw', 'label': 'Wind'},
                   {'name': 'solar_capacity_mw', 'label': 'Solar'},
                   {'name': 'thermal_capacity_mw', 'label': 'Thermal'}]
    y_variable_axis = 'Capacity (MW)'
    title = 'Optimal Capacities (MW)'
    output_path = folder_path / 'optimal_capacities.png'
    line_plot(events_data, events, y_variables, y_variable_axis, X_VARIABLE, output_path, title)

            
def format_conditional_means(folder_path: Path) -> pd.DataFrame:
    def header(name):
        return pd.DataFrame([
            ['-' * 10 + f' {name} ' + '-' * 10],  # Header with dashes
            ['']  # Blank line
        ])
    # Load the event data
    events_data = events_data_from_csv(folder_path / 'conditional_means.csv')

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

def generate_colot_palette(total_lines: int):

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

    return colors



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

    # Calculate total number of lines needed (events × variables)
    total_lines = len(events) * len(y_variables)

    # Generate a color palette
    colors = generate_colot_palette(total_lines)
    
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
