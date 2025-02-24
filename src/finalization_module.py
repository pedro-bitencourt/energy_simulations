import logging
from pathlib import Path
import subprocess
import pandas as pd


from .utils.load_configs import load_events, load_plots, load_plots_r, load_comparisons
from .plotting_module import event_comparison_plot, line_plot
from .constants import BASE_PATH

logger = logging.getLogger(__name__)

PATH_R_SCRIPT: Path = BASE_PATH / 'code/r/run_analysis.R'
GRAPHICS_FOLDER: Path = BASE_PATH / 'figures'

class SimulationFinalizer:
    def __init__(self, simulation_folder: Path, x_variable: dict, 
                 participants: list[str] = ['wind', 'solar', 'thermal', 'salto'], overwrite: bool = False):
        # Save inputs
        self.simulation_folder: Path = simulation_folder
        self.x_variable: dict[str, str] = x_variable
        self.participants: list[str] = participants

        self.capacities_variables: list[str] = [f'{participant}_capacity' 
            for participant in participants]

        # Create output folder
        self.output_folder: Path = GRAPHICS_FOLDER / simulation_folder.name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Load data
        self.capacities_df: pd.DataFrame = pd.read_csv(simulation_folder /
            'results'/'solver_results.csv')[self.capacities_variables + [x_variable['name'], 'name']]
        logger.info("Capacities data: %s", self.capacities_df.head())
        self.conditional_means_df: pd.DataFrame = pd.read_csv(simulation_folder /
            'results'/'conditional_means.csv')
        logger.info("Conditional means data: %s", self.conditional_means_df.head())
        
        self.solver_results_df: pd.DataFrame = pd.read_csv(simulation_folder /
            'results'/'solver_results.csv')

        
        # Process data
        self.events_data: dict[str, pd.DataFrame] = self.events_data_from_csv()
        self.results_summary: pd.DataFrame = self.get_results_summary(overwrite)
        logger.info("Results summary: %s", self.results_summary.head())

    def get_results_summary(self, overwrite: bool = False) -> pd.DataFrame:
        if not overwrite and (self.simulation_folder / 'results_summary.csv').exists():
            logger.info("Results already exist for simulation %s. Skipping R script.",
                        self.simulation_folder.name)
            return pd.read_csv(self.simulation_folder / 'results_summary.csv')
        logger.info("Running R script for simulation %s", self.simulation_folder.name)
        cmd = f"Rscript {PATH_R_SCRIPT} {self.simulation_folder.name}"
        subprocess.run(cmd, shell=True, check=True)
        results_summary: pd.DataFrame =  pd.read_csv(self.simulation_folder / 'results_summary.csv')
        # Merge with capacities
        results_summary = results_summary.merge(self.solver_results_df, on='name', how='outer')
        # Order by x_variable
        results_summary = results_summary.sort_values(by=self.x_variable['name'])
        return results_summary

    def events_data_from_csv(self) -> dict[str, pd.DataFrame]:
        """
        Load the data from conditional_means.csv into a dictionary of DataFrames,
        one for each event.
        """
        logger.info("Starting the events_data_from_csv() function at path: %s",
                    str(self.simulation_folder))

        # Load the data
        events_data = events_dataframes(self.conditional_means_df)

        # Remove the prefixes
        for event, df in events_data.items():
            df.columns = [col.replace(f"{event}_", "") for col in df.columns]

            # Merge df with capacities
            df = pd.concat([df, self.capacities_df], axis=1)

            df[self.x_variable['name']] = self.conditional_means_df[self.x_variable['name']]

            logger.debug("Dataframe for event %s columns after processing: %s", event, df.columns)
            events_data[event] = df
        return events_data

    def sim_line_plot(self, y_variables: dict[str, dict[str, str]], y_variable_axis: str, title: str, output_path: Path):
        line_plot(self.results_summary,
                  self.x_variable['name'],
                  y_variables,
                  title,
                  self.x_variable['label'],
                  y_variable_axis,
                  output_path)


def finalize(simulation: SimulationFinalizer):
    # Call the formatting functions
    format_results(simulation)

    # Call the plotting functions
    visualize(simulation)


def visualize(simulation: SimulationFinalizer):
    """
    Generate visualizations from the simulation results.
    """
    logger.info("Starting the visualize() function at path: %s", str(simulation.simulation_folder))

    # Call plotting functions
    plot_event_comparisons(simulation)
    plot_optimal_capacities(simulation)
    #plot_std_revenues(simulation)
    plot_results_summary(simulation)



def plot_event_comparisons(simulation: SimulationFinalizer):
    logger.info("Starting the plot_event_comparisons() function at path: %s",
                str(simulation.simulation_folder))

    # Create comparisons folder
    (simulation.simulation_folder / 'graphics/comparisons').mkdir(parents=True, exist_ok=True)

    logger.info("Plotting event comparisons...")
    for comparison_name, set_events in load_comparisons().items():
        logger.info("Plotting comparison: %s", comparison_name)
        for plot_name, plot_config in load_plots().items():
            set_events_temp = set_events
            if plot_name == 'frequency':
                # remove 'unconditional' event
                set_events_temp = [event for event in set_events if event != 'unconditional']

            comparison_folder = (simulation.output_folder/ 'comparisons' / comparison_name)
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"
            title = f"{plot_config['title']} - {comparison_name}"

            logger.debug("Plotting %s", title)
            event_comparison_plot(simulation.events_data, set_events_temp, plot_config['variables'],
                      plot_config['y_label'], simulation.x_variable, file_path,
                      title)

def plot_results_summary(simulation: SimulationFinalizer):
    # Load the plots configuration
    plots_r = load_plots_r()

    for plot_name, plot in plots_r.items():
        title = ""
        output_path = simulation.output_folder / f"{plot_name}.png"
        y_variables = {variable['name']: {'label' : variable['label'],
                                          'color': variable['color']}
            for variable in plot['variables']}
        line_plot(simulation.results_summary,
            simulation.x_variable['name'],
            y_variables,
            title,
            simulation.x_variable['label'],
            plot['y_label'],
            output_path)

def plot_optimal_capacities(simulation: SimulationFinalizer):
    # Load the investment simulation_folder
    color_map = {'wind': '#2CA02C', 'solar': '#FFD600', 'salto': '#1F77B4', 'thermal': '#D62728'}
    y_variables = {f'{participant}_capacity': {
        'label' : f'{participant} Capacity (MW)',
        'color': color_map[participant]} for participant in simulation.participants}     
    y_variable_axis = 'Capacity (MW)'
    title = 'Optimal Capacities (MW)'
    output_path = simulation.output_folder / 'optimal_capacities.png'
    line_plot(simulation.solver_results_df,
              simulation.x_variable['name'],
              y_variables,
              title,
              simulation.x_variable['label'],
              y_variable_axis, output_path)


def plot_std_revenues(simulation: SimulationFinalizer):
    # Divide the revenue standard deviation by the capacity
    for participant in simulation.participants:
        if participant != 'salto':
            std_rev_col = f'revenue_{participant}_std'
            std_rev_col_new = f'revenue_{participant}_std_per_mw'
            capacity_col = f'{participant}_capacity'
            simulation.solver_results_df[std_rev_col_new] = simulation.solver_results_df[std_rev_col] / capacities[capacity_col]

    # Define the variables to plot
    color_map = {'wind': '#2CA02C', 'solar': '#FFD600', 'salto': '#1F77B4', 'thermal': '#D62728'}
    y_variables = {f'revenue_{participant}_std_per_mw': {
        'label' : f'{participant}',
        'color': color_map[participant]} for participant in simulation.participants
    }     
    y_variable_axis = 'Standard Deviation of Expected Profitts ($)'
    title = 'Standard Deviation of Expected Profits ($)'
    output_path = simulation.output_folder / 'std_profits.png'

    line_plot(simulation.solver_results, x_variable['name'], y_variables, title, x_variable['label'], y_variable_axis, output_path)
            

####################################################################################################
# Formatting functions
####################################################################################################
def format_conditional_means(simulation: SimulationFinalizer) -> pd.DataFrame:
    def header(name):
        return pd.DataFrame([
            ['-' * 10 + f' {name} ' + '-' * 10],  # Header with dashes
            ['']  # Blank line
        ])
    events_data: dict[str, pd.DataFrame] = simulation.events_data
    x_variable: dict[str, str] = simulation.x_variable
    # Prepare the output DataFrame
    final_df = pd.DataFrame()

    # Loop through each event
    EVENTS, _ = load_events()
    for event, event_data in EVENTS.items():
        # Add event header and append to final output
        if event not in events_data:
            logger.warning("Event %s not found in events_data. Skipping.", event)
            continue
        event_header = header(event_data)
        # Order in the x variable
        event_data = event_data.sort_values(by=x_variable['name'])
        final_df = pd.concat([final_df, event_header, event_data], ignore_index=True)
        # Add a blank row as separator
        blank_row = pd.DataFrame({col: None for col in final_df.columns}, index=[0])
        final_df = pd.concat([final_df, blank_row], ignore_index=True)

    return final_df

def format_results(simulation: SimulationFinalizer):
    # Format csv to be used in the report
    formatted_df = format_conditional_means(simulation)
    formatted_df.to_csv(simulation.simulation_folder / 'formatted_conditional_means.csv', index=False)

    # Create and plot frequencies table
    frequencies_table = build_frequencies_table(simulation)
    frequencies_table.to_csv(simulation.simulation_folder / 'frequencies_table.csv', index=False)

def build_frequencies_table(simulation: SimulationFinalizer) -> pd.DataFrame:
    """
    Creates a table where each event's 'frequency' column is aligned with the x_variable.

    Args:
        simulation_folder (Path): Path to the folder with simulation data (
        e.g., 'conditional_means.csv').
        x_variable (dict): Dictionary with the x-axis variable, e.g., {'name': 'salto_capacity'}.

    Returns:
        pd.DataFrame: Table with columns [x_variable['name'], event_1, event_2, ...].
    """
    logger.info("Building frequencies table...")

    events_data = simulation.events_data
    x_variable = simulation.x_variable

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

####################################################################################################
# Parsing functions
def events_dataframes(conditional_means_df: pd.DataFrame) -> dict:
    EVENTS, _ = load_events()
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
