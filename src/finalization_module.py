import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
import pandas as pd
import logging

from .utils.load_configs import load_events, load_plots, load_comparisons
from .data_analysis_module import PARTICIPANTS, VARIABLES
from .plotting_module import event_comparison_plot, line_plot

logger = logging.getLogger(__name__)

# Load the events, variables, plots and comparisons
EVENTS: dict = load_events()
VARIABLES_TO_PLOT: dict = load_plots()
COMPARISON_EVENTS: dict = load_comparisons()

VARIABLES = [
    *[f'production_{participant}' for participant in PARTICIPANTS],
    *[f'variable_cost_{participant}' for participant in PARTICIPANTS],
    #*[f'revenue_{participant}' for participant in PARTICIPANTS],
    #*[f'profit_{participant}' for participant in PARTICIPANTS],
    'marginal_cost',
    'demand',
    'frequency'
]
# TO FIX
CAPACITY_VARIABLES = [
    *[f'{participant}_capacity' for participant in PARTICIPANTS]
]


# This dictionary will store the events to be plotted, along with their labels
# Load events from events.json

# Include x_variable in CAPACITY_VARIABLES

def visualize(results_folder: Path, x_variable: dict):
    """
    Generate visualizations from the simulation results.

    Args:

    x_variable: dict = {
        'name': 'salto_capacity', 'label': 'Salto Capacity (MW)'
        #'name': 'lake_factor', 'label': 'Lake Factor'
    }
    """
    logger.info("Starting the visualize() function at path: " + str(results_folder))

    paths: dict[str, Path] = {
        'graphics': results_folder / 'graphics',
        'conditional_means': results_folder / 'conditional_means.csv',
        'investment_results': results_folder / 'investment_results.csv'
    }

    # Create relevant folders
    for path in paths.values():
        # Check if path is dir
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

    # Visualize conditional means
    plot_event_comparisons(results_folder, x_variable)

    # Format csv
    formatted_df = format_conditional_means(results_folder, x_variable)

    # Save to disk
    formatted_df.to_csv(results_folder / 'formatted_conditional_means.csv', index=False)

    # Plot optimal capacities
    plot_optimal_capacities(results_folder, x_variable)

    # Plot standard deviation of revenues
    plot_std_revenues(results_folder, x_variable)
    pass


def events_dataframes(conditional_means_df: pd.DataFrame):
    events_data: dict = {}
    for event in EVENTS.keys():
        try:
            # Check if all columns exist for the event
            columns = [f'{event}_{var}' for var in VARIABLES]
            missing_columns = [col for col in columns if col not in conditional_means_df.columns]
    
            if missing_columns:
                logger.warning("Event %s not found or missing columns: %s", event, missing_columns)
                continue
    
            # If columns are found, add the event data to the dictionary
            events_data[event] = conditional_means_df[columns]
    
        except Exception as e:
            logger.warning("An unexpected error occurred while processing event %s: %s", event, str(e))
            continue
    return events_data

def events_data_from_csv(folder_path: Path, x_variable: dict) -> dict[str, pd.DataFrame]:
    logger.info("Starting the events_data_from_csv() function at path: " + str(folder_path))
    # Load the conditional means dataframe
    conditional_means_df = pd.read_csv(folder_path / 'conditional_means.csv')

    logger.info("Conditional means dataframe loaded: %s", conditional_means_df.head())
    print(conditional_means_df)

    logger.debug("EVENTS: %s", EVENTS)
    logger.debug("VARIABLES: %s", VARIABLES)

    # Create a dictionary with the data for each event
    events_data = events_dataframes(conditional_means_df)


    # Merge CAPACITY_VARIABLES with x_variable
    #capacities_df = conditional_means_df[[x_variable['name']] + CAPACITY_VARIABLES]
    capacities_df = conditional_means_df[CAPACITY_VARIABLES]

    logger.info("Events data loaded: %s", events_data)

    # Remove the prefixes
    for event, df in events_data.items():
        df.columns = [col.replace(f"{event}_", "") for col in df.columns]

        # Merge df with capacities
        df = pd.concat([df, capacities_df], axis=1)
        # Get utilization rate for thermal and hydro
        df['utilization_thermal'] = df['production_thermal'] / df['thermal_capacity']
        df['utilization_salto'] = df['production_salto'] / df['salto_capacity']

        logger.debug("Dataframe for event %s columns after processing: %s", event, df.columns)
        events_data[event] = df
    return events_data

def plot_event_comparisons(folder_path: Path, x_variable: dict):
    logger.info("Starting the plot_event_comparisons() function at path: " + str(folder_path))
    # Load events data from csv
    events_data = events_data_from_csv(folder_path, x_variable)


    logger.info("Plotting event comparisons...")
    for comparison_name, set_events in COMPARISON_EVENTS.items():
        logger.info("Plotting comparison: %s", comparison_name)
        for plot_name, plot_config in VARIABLES_TO_PLOT.items():
            if plot_name == 'production':
                # remove unconditional
                set_events_temp = [event for event in set_events if event != 'unconditional']
                if not set_events_temp:
                    set_events_temp = set_events
            else:
                set_events_temp = set_events
            print(f"{set_events_temp=}")
            comparison_folder = folder_path / comparison_name
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"
            title = f"{plot_config['title']} - {comparison_name}"

            logger.info("Plotting %s", title)
            event_comparison_plot(events_data, set_events_temp, plot_config['variables'],
                      plot_config['y_label'], x_variable, file_path,
                      title)

def load_investment_results(folder_path: Path) -> pd.DataFrame:
    return pd.read_csv(folder_path / 'investment_results.csv')

def plot_optimal_capacities(folder_path: Path, x_variable: dict):
    # Load the investment results_folder
    investment_results = load_investment_results(folder_path)
    y_variables = {f'{participant}_capacity': f'{participant} Capacity (MW)' for participant in PARTICIPANTS}
    y_variable_axis = 'Capacity (MW)'
    title = 'Optimal Capacities (MW)'
    output_path = folder_path / 'optimal_capacities.png'
    line_plot(investment_results, x_variable['name'], y_variables, title, x_variable['label'], y_variable_axis, output_path)

def plot_std_revenues(folder_path: Path, x_variable: dict):
    # Load the investment results_folder
    investment_results = load_investment_results(folder_path)
    y_variables = {f'{participant}_std_revenue': f'{participant} Capacity (MW)' for participant in PARTICIPANTS}
    y_variable_axis = 'Standard Deviation of Expected Profitts ($)'
    title = 'Standard Deviation of Expected Profits ($)'
    output_path = folder_path / 'std_profits.png'
    # Check if the columns exist
    y_variables_new = {}
    for y_var in y_variables.keys():
        if y_var not in investment_results.columns:
            logger.warning("Column %s not found in investment_results. Skipping.", y_var)
            continue
        y_variables_new[y_var] = y_variables[y_var]

    line_plot(investment_results, x_variable['name'], y_variables_new, title, x_variable['label'], y_variable_axis, output_path)
            
def format_conditional_means(folder_path: Path, x_variable) -> pd.DataFrame:
    def header(name):
        return pd.DataFrame([
            ['-' * 10 + f' {name} ' + '-' * 10],  # Header with dashes
            ['']  # Blank line
        ])
    # Load the event data
    events_data: dict = events_data_from_csv(folder_path, x_variable)


    # Prepare the output DataFrame
    final_df = pd.DataFrame()

    # Loop through each event
    for event in EVENTS.keys():
        # Add event header and append to final output
        if event not in events_data:
            logger.warning("Event %s not found in events_data. Skipping.", event)
            continue
        event_header = header(EVENTS[event])
        event_data = events_data[event]
        print(f"{event_data.columns=}")
        print(f"{event_data['frequency'].head()=}")
        print(f"{event_data.head()=}")
        # Order in salto_capacity
        event_data = event_data.sort_values(by=x_variable['name'])
        final_df = pd.concat([final_df, event_header, event_data], ignore_index=True)
        # Add a blank row as separator
        blank_row = pd.DataFrame({col: None for col in final_df.columns}, index=[0])
        final_df = pd.concat([final_df, blank_row], ignore_index=True)

    return final_df

