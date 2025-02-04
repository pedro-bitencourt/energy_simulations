import logging
from pathlib import Path
import subprocess
import pandas as pd


from .utils.load_configs import load_events, load_plots, load_plots_r, load_comparisons
from .data_analysis_module import PARTICIPANTS
from .plotting_module import event_comparison_plot, line_plot

logger = logging.getLogger(__name__)

PATH_R_SCRIPT: Path = Path('/Users/pedrobitencourt/Projects/energy_simulations/code/r/run_analysis.R')
GRAPHICS_FOLDER: Path = Path('/Users/pedrobitencourt/Projects/energy_simulations/figures')

# TO FIX
CAPACITY_VARIABLES = [
    *[f'{participant}_capacity' for participant in PARTICIPANTS]
]

def finalize(simulation_folder: Path, x_variable: dict):
    # Call the plotting functions
    visualize(simulation_folder, x_variable)

    # Call the formatting functions
    format_results(simulation_folder, x_variable)

    # Run the R script
    finalize_r(simulation_folder, x_variable)

def finalize_r(simulation_folder: Path, x_variable: dict):
    # Run the R script on the simulation folder
    logger.info("Running R script on the simulation folder...")
    cmd = f"Rscript {PATH_R_SCRIPT} {simulation_folder}"
    subprocess.run(cmd, shell=True, check=True)

    # Plot the results summary
    plot_results_summary(simulation_folder, x_variable)

def format_results(simulation_folder: Path, x_variable: dict):
    # Format csv to be used in the report
    formatted_df = format_conditional_means(simulation_folder, x_variable)
    formatted_df.to_csv(simulation_folder / 'formatted_conditional_means.csv', index=False)

    # Create and plot frequencies table
    frequencies_table = build_frequencies_table(simulation_folder, x_variable)
    frequencies_table.to_csv(simulation_folder / 'frequencies_table.csv', index=False)


def build_frequencies_table(simulation_folder: Path, x_variable: dict) -> pd.DataFrame:
    """
    Creates a table where each event's 'frequency' column is aligned with the x_variable.

    Args:
        simulation_folder (Path): Path to the folder with simulation data (e.g., 'conditional_means.csv').
        x_variable (dict): Dictionary with the x-axis variable, e.g., {'name': 'salto_capacity'}.

    Returns:
        pd.DataFrame: Table with columns [x_variable['name'], event_1, event_2, ...].
    """
    logger.info("Building frequencies table...")
    events_data = events_data_from_csv(simulation_folder)

    # Build and merge frequency columns for all events
    combined_df = pd.concat(
        [
            df[[x_variable['name'], 'frequency']].rename(columns={'frequency': event_name})
            for event_name, df in events_data.items()
        ],
        axis=1,
    )

    # Drop duplicate x_variable columns created during concatenation
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Sort by the x_variable for clarity
    return combined_df.sort_values(by=x_variable['name'])


def visualize(simulation_folder: Path, x_variable: dict):
    """
    Generate visualizations from the simulation results.

    Args:

    x_variable: dict = {
        'name': 'salto_capacity', 'label': 'Salto Capacity (MW)'
        #'name': 'lake_factor', 'label': 'Lake Factor'
    }
    """
    logger.info("Starting the visualize() function at path: " + str(simulation_folder))

    # Create a folder for the graphics
    simulation_folder = Path(simulation_folder)
    Path(simulation_folder / 'graphics').mkdir(parents=True, exist_ok=True)

    # Call plotting functions
    plot_event_comparisons(simulation_folder, x_variable)
    plot_optimal_capacities(simulation_folder, x_variable)
    plot_std_revenues(simulation_folder, x_variable)

def events_dataframes(conditional_means_df: pd.DataFrame) -> dict:
    EVENTS = load_events()
    # Derive variables dynamically from 'unconditional' columns
    variables = [
        col[len("unconditional_"):]
        for col in conditional_means_df.columns
        if col.startswith("unconditional_")
    ]
    
    events_data = {}
    for event in EVENTS.keys():
        # Build expected columns for this event
        event_cols = [f"{event}_{var}" for var in variables]
        
        # Ensure all required columns exist
        if all(col in conditional_means_df.columns for col in event_cols):
            events_data[event] = conditional_means_df[event_cols]
        else:
            logger.warning(
                "Event '%s' missing some columns: %s",
                event,
                [col for col in event_cols if col not in conditional_means_df.columns],
            )
    
    return events_data

def load_capacities(simulation_folder) -> pd.DataFrame:
    return pd.read_csv(simulation_folder / 'conditional_means.csv')[CAPACITY_VARIABLES]

def load_conditional_mean(simulation_folder) -> pd.DataFrame:
    return pd.read_csv(simulation_folder / 'conditional_means.csv')

def events_data_from_csv(simulation_folder: Path) -> dict[str, pd.DataFrame]:
    logger.info("Starting the events_data_from_csv() function at path: " + str(simulation_folder))
    # Load the data
    conditional_means_df = load_conditional_mean(simulation_folder)
    events_data = events_dataframes(conditional_means_df)
    capacities_df = load_capacities(simulation_folder)

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

def plot_event_comparisons(simulation_folder: Path, x_variable: dict):
    logger.info("Starting the plot_event_comparisons() function at path: " + str(simulation_folder))
    # Load events data from csv
    events_data = events_data_from_csv(simulation_folder)

    # Create comparisons folder
    (simulation_folder / 'graphics/comparisons').mkdir(parents=True, exist_ok=True)

    logger.info("Plotting event comparisons...")
    for comparison_name, set_events in load_comparisons().items():
        logger.info("Plotting comparison: %s", comparison_name)
        for plot_name, plot_config in load_plots().items():
            set_events_temp = set_events
            if plot_name == 'frequency':
                # remove 'unconditional' event
                set_events_temp = [event for event in set_events if event != 'unconditional']

            comparison_folder = simulation_folder / 'graphics' / 'comparisons' / comparison_name
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"
            title = f"{plot_config['title']} - {comparison_name}"

            logger.info("Plotting %s", title)
            event_comparison_plot(events_data, set_events_temp, plot_config['variables'],
                      plot_config['y_label'], x_variable, file_path,
                      title)

def load_investment_results(simulation_folder: Path) -> pd.DataFrame:
    return pd.read_csv(simulation_folder / 'investment_results.csv')


def plot_results_summary(simulation_folder: Path, x_variable: dict):
    # Load investment results
    investment_results = load_investment_results(simulation_folder)
    # Load the results summary
    results_summary = pd.read_csv(simulation_folder / "results_summary.csv")
    # Merge the two on 'name'
    results_summary = results_summary.merge(investment_results, on='name', how='outer')
    # Order by x_variable
    results_summary = results_summary.sort_values(by=x_variable['name'])

    # Load the plots configuration
    plots_r = load_plots_r()

    # Get the graphics folder
    graphics_folder: Path = get_graphics_folder(simulation_folder)

    for plot_name, plot in plots_r.items():
        title = ""
        output_path = graphics_folder / f"{plot_name}.png"
        y_variables = {variable['name']: {'label' : variable['label'], 'color': variable['color']} for variable in plot['variables']}
        line_plot(results_summary,
            x_variable['name'],
            y_variables,
            title,
            x_variable['label'],
            plot['y_label'],
            output_path)


def get_graphics_folder(simulation_folder: Path) -> Path:
    graphics_folder: Path = GRAPHICS_FOLDER / simulation_folder.name
    graphics_folder.mkdir(parents=True, exist_ok=True)
    return graphics_folder
    

def plot_optimal_capacities(simulation_folder: Path, x_variable: dict):
    # Load the investment simulation_folder
    investment_results = load_investment_results(simulation_folder)
    color_map = {'wind': '#2CA02C', 'solar': '#FFD600', 'salto': '#1F77B4', 'thermal': '#D62728'}
    y_variables = {f'{participant}_capacity': {
        'label' : f'{participant} Capacity (MW)',
        'color': color_map[participant]} for participant in PARTICIPANTS
    }     
    y_variable_axis = 'Capacity (MW)'
    title = 'Optimal Capacities (MW)'
    output_path = get_graphics_folder(simulation_folder) / 'optimal_capacities.png'
    line_plot(investment_results, x_variable['name'], y_variables, title, x_variable['label'], y_variable_axis, output_path)

def plot_std_revenues(simulation_folder: Path, x_variable: dict):
    # Load the investment simulation_folder
    investment_results = load_investment_results(simulation_folder)

    capacities = load_capacities(simulation_folder)

    # Divide the revenue standard deviation by the capacity
    for participant in PARTICIPANTS:
        if participant != 'salto':
            investment_results[f'revenue_{participant}_std_new'] = investment_results[f'revenue_{participant}_std'] / capacities[f'{participant}_capacity']

    # Define the variables to plot
    color_map = {'wind': '#2CA02C', 'solar': '#FFD600', 'salto': '#1F77B4', 'thermal': '#D62728'}
    y_variables = {f'revenue_{participant}_std_new': {
        'label' : f'{participant}',
        'color': color_map[participant]} for participant in PARTICIPANTS
    }     
    y_variable_axis = 'Standard Deviation of Expected Profitts ($)'
    title = 'Standard Deviation of Expected Profits ($)'
    output_path = get_graphics_folder(simulation_folder) / 'std_profits.png'

    line_plot(investment_results, x_variable['name'], y_variables, title, x_variable['label'], y_variable_axis, output_path)
            
def format_conditional_means(simulation_folder: Path, x_variable) -> pd.DataFrame:
    def header(name):
        return pd.DataFrame([
            ['-' * 10 + f' {name} ' + '-' * 10],  # Header with dashes
            ['']  # Blank line
        ])
    # Load the event data
    events_data: dict = events_data_from_csv(simulation_folder)

    # Prepare the output DataFrame
    final_df = pd.DataFrame()

    # Loop through each event
    EVENTS = load_events()
    for event in EVENTS.keys():
        # Add event header and append to final output
        if event not in events_data:
            logger.warning("Event %s not found in events_data. Skipping.", event)
            continue
        event_header = header(EVENTS[event])
        event_data = events_data[event]
        # Order in salto_capacity
        event_data = event_data.sort_values(by=x_variable['name'])
        final_df = pd.concat([final_df, event_header, event_data], ignore_index=True)
        # Add a blank row as separator
        blank_row = pd.DataFrame({col: None for col in final_df.columns}, index=[0])
        final_df = pd.concat([final_df, blank_row], ignore_index=True)

    return final_df

