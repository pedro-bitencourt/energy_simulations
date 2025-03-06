import logging
from pathlib import Path
import pandas as pd

from typing import Any

from .comparative_statics_module import load_costs
from .run_analysis_module import analyze_run

from .utils.load_configs import load_events, load_plots, load_plots_r, load_comparisons
from .utils.auxiliary import skip_if_exists
from .plotting_module import event_comparison_plot, line_plot
from .constants import BASE_PATH


logger = logging.getLogger(__name__)

NROWS_TO_READ = 500_000

GRAPHICS_FOLDER: Path = BASE_PATH / 'figures'

class SimulationFinalizer:
    def __init__(self,
                 simulation_folder: Path,
                 x_variable: dict[str, Any],
                 endogenous_participants: list[str],
                 costs_path: Path,
                 overwrite: bool = False):
        # Get name of the experiment
        self.name: str = simulation_folder.name

        # Save inputs
        self.x_variable: dict[str, Any] = x_variable
        self.endogenous_participants: list[str] = endogenous_participants


        # Set up paths
        self.paths: dict[str, Path] = self._construct_paths(simulation_folder)

        # Create output folder
        self.paths['output_folder'].mkdir(parents=True, exist_ok=True)

        self.exogenous_values = self.get_exogenous_values()

        # Load data
        self.capacities_df: pd.DataFrame = self.load_capacities()
        logger.debug("Capacities data: %s", self.capacities_df.head())

        costs_dict: dict[str, dict[str, float]] = load_costs(costs_path)
        self.fixed_cost_dict: dict[str, float] = costs_dict["fixed_cost_dictionary"]
        self.marginal_cost_dict: float = costs_dict["marginal_cost_dictionary"]

        # Process data
        self.events_data: dict[str, pd.DataFrame] = self.events_data_from_csv()
        self.results: pd.DataFrame = self.get_results(overwrite)
        logger.debug("Results: %s", self.results.head())

    def get_exogenous_values(self):
        # Grab run files
        run_files: list[Path] = list(self.paths['raw'].glob('*.csv'))
        exogenous_values: list[dict[str, Any]] = [{  
            "value": file.stem.split('_')[-1],
            "path": file,
            "name": file.stem
        } for file in run_files]
        return exogenous_values


    def load_capacities(self):
        capacities_variables: list[str] = [f'{participant}_capacity'
            for participant in self.endogenous_participants]
        solver_results_df: pd.DataFrame = pd.read_csv(self.paths['solver_results'])
        capacities_df: pd.DataFrame = solver_results_df[capacities_variables + [self.x_variable['name'], 'name']]
        return capacities_df

    def _construct_paths(self, simulation_folder: Path):
        paths: dict[str, Path] = {
            'simulation_folder': simulation_folder,
            'raw': simulation_folder / 'raw',
            'output_folder': GRAPHICS_FOLDER / self.name,
            'formatted_conditional_means': simulation_folder / 'formatted_conditional_means.csv',
            'frequencies_table': simulation_folder / 'frequencies_table.csv',
            'results': simulation_folder / 'results.csv',
            'solver_results': simulation_folder / 'results' / 'solver_results.csv',
            'conditional_means': simulation_folder / 'results' / 'conditional_means.csv'
        }
        return paths

    def get_results(self, overwrite: bool) -> pd.DataFrame:
        if skip_if_exists(self.paths['results'], overwrite):
            return pd.read_csv(self.paths['results'])
        results_df: pd.DataFrame = analyze_runs(self)
        results_df = results_df.sort_values(by=self.x_variable['name'])
        results_df.to_csv(self.paths['results'], index=False)
        return results_df

    def events_data_from_csv(self) -> dict[str, pd.DataFrame]:
        """
        Load the data from conditional_means.csv into a dictionary of DataFrames,
        one for each event.
        """
        logger.info("Starting the events_data_from_csv() function at path: %s",
                    str(self.paths['simulation_folder']))
        # Load the data
        conditional_means_df: pd.DataFrame = pd.read_csv(self.paths['simulation_folder'] / 'results' / 'conditional_means.csv')
        events_data = events_dataframes(conditional_means_df)
        # Remove the prefixes
        for event, df in events_data.items():
            df.columns = [col.replace(f"{event}_", "") for col in df.columns]
            # Merge df with capacities
            df = pd.concat([df, self.capacities_df], axis=1)
            df[self.x_variable['name']] = conditional_means_df[self.x_variable['name']]
            logger.debug("Dataframe for event %s columns after processing: %s", event, df.columns)
            events_data[event] = df
        return events_data

    def sim_line_plot(self, y_variables: dict[str, dict[str, str]], y_variable_axis: str, title: str, output_path: Path):
        line_plot(self.results,
                  self.x_variable['name'],
                  y_variables,
                  title,
                  self.x_variable['label'],
                  y_variable_axis,
                  output_path)

        

def analyze_runs(simulation: SimulationFinalizer):
    results: list[dict[str, Any]] = []
    exogenous_values = simulation.exogenous_values
    capacities_df: pd.DataFrame = simulation.capacities_df
    for exogenous_value_dict in exogenous_values:
        print("Analyzing exogenous value ", exogenous_value_dict)
        run_name = exogenous_value_dict['name']
        exogenous_value = exogenous_value_dict['value']
        input_path = exogenous_value_dict['path']
    
        run_capacities = capacities_df.loc[capacities_df["name"] == run_name, ]
        if run_capacities.empty:
            logger.warning("No capacities found for run %s. Skipping.", run_name)
            continue
        run_data: pd.DataFrame = pd.read_csv(input_path) #, nrows= NROWS_TO_READ)
        output_folder = simulation.paths['output_folder'] / "runs" / run_name
        output_folder.mkdir(parents=True, exist_ok=True)
        results_run = analyze_run(
            data=run_data,
            run_name=run_name,
            output_folder=output_folder,
            exogenous_variable_value=exogenous_value,
            run_capacities=run_capacities,
            fixed_cost_dict=simulation.fixed_cost_dict, 
            marginal_cost_dict=simulation.marginal_cost_dict,
            overwrite=True
        )
        results.append(results_run)

        logger.critical(results_run)
    
    results_df: pd.DataFrame = pd.DataFrame(results)
    results_df.to_csv(simulation.paths['results'], index=False)
    return results_df


def finalize(simulation: SimulationFinalizer):
    format_results(simulation)
    visualize(simulation)


def visualize(simulation: SimulationFinalizer, overwrite: bool = False):
    """
    Generate visualizations from the simulation results.
    """
    logger.info("Starting the visualize() function at path: %s", str(simulation.paths['simulation_folder']))

    plot_event_comparisons(simulation, overwrite)
    plot_results(simulation)

def plot_event_comparisons(simulation: SimulationFinalizer, overwrite: bool = False):
    logger.info("Starting the plot_event_comparisons() function at path: %s",
                str(simulation.paths['simulation_folder']))

    # Create comparisons folder
    (simulation.paths['simulation_folder'] / 'graphics/comparisons').mkdir(parents=True, exist_ok=True)

    logger.info("Plotting event comparisons...")
    for comparison_name, set_events in load_comparisons().items():
        logger.info("Plotting comparison: %s", comparison_name)
        for plot_name, plot_config in load_plots().items():
            set_events_temp = set_events
            if plot_name == 'frequency':
                # remove 'unconditional' event
                set_events_temp = [event for event in set_events if event != 'unconditional']

            comparison_folder = (simulation.paths['output_folder']/ 'comparisons' / comparison_name)
            # Create folder
            comparison_folder.mkdir(parents=True, exist_ok=True)
            file_path = comparison_folder /f"{plot_name}.png"

            if skip_if_exists(file_path, overwrite):
                continue
            title = f"{plot_config['title']} - {comparison_name}"

            logger.debug("Plotting %s", title)
            event_comparison_plot(simulation.events_data, set_events_temp, plot_config['variables'],
                      plot_config['y_label'], simulation.x_variable, file_path,
                      title)

def plot_results(simulation: SimulationFinalizer):
    # Load the plots configuration
    plots_r = load_plots_r()

    for plot_name, plot in plots_r.items():
        title = ""
        output_path = simulation.paths['output_folder'] / f"{plot_name}.png"
        y_variables = {variable['name']: {'label' : variable['label'],
                                          'color': variable['color']}
            for variable in plot['variables']}

        # Order results by x_variable
        simulation.results = simulation.results.sort_values(by=simulation.x_variable['name'])
        line_plot(simulation.results,
            simulation.x_variable['name'],
            y_variables,
            title,
            simulation.x_variable['label'],
            plot['y_label'],
            output_path)

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

