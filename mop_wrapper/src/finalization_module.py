"""
This module contains functions to process and plot the data extracted from MOP.

The main object is the SimulationData class, which stores all the relevant data produced by 
the ComparativeStatics class. It contains the name of the simulation, the run data, the 
participants, the x variable, and the cost parameters. 

The main functions are: 
- build_simulation_data: Construct a SimulationData instance by reading all CSV files in runs_folder.
- default_analysis: Perform the default analysis on the simulation data (that is, using the 
configurations specified in the config.yaml file and in the run_analysis_module).
- plot_results: Plot the results of the simulation.
- plot_densities: Plot the densities of selected variables.
"""

from typing import Iterator, Tuple, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from functools import partial
import logging
from matplotlib.figure import Figure

from .utils.auxiliary import log_exceptions, log_execution_time

from .constants import BASE_PATH
from .run_analysis_module import full_run_analysis
from .plotting_module import line_plot
from .utils.load_configs import load_plot_configs, load_costs, load_events_config
from .utils.statistics_utils import plot_densities_run



logger = logging.getLogger(__name__)

# Custom types
"""A RunData tuple contains a dataframe and additional information (e.g., capacities)
in the form of a dictionary."""
RunData = Tuple[pd.DataFrame, Dict[str, Any]]
RunAnalysisFunction = Callable[[RunData], Dict[str, Any]]


@dataclass(frozen=True)
class SimulationData:
    """
    The class SimulationData stores all the relevant data produced by the 
    ComparativeStatics class.
    
    Attributes:
        - name: the name of the simulation.
        - run_data: a list of tuples, each containing the run name, its dataframe,
          and additional information (e.g., capacities).
        - participants: a list of the participants in the simulation.
        - x_variable: a dictionary containing the name and label of the x variable.
        - cost_parameters: a dictionary containing the cost parameters used in the simulation.
    """
    name: str
    run_data: Iterator[RunData]
    participants: List[str]
    x_variable: Dict[str, Any]
    cost_parameters: Dict[str, Any]


def build_run_data(file: Path, solver_results: pd.DataFrame, test: bool = False) -> RunData:
    """
    Given a CSV file and the solver_results dataframe, construct a RunData tuple.
    The run name is derived from the file stem.
    """
    if test:
        df = pd.read_csv(file, nrows=1000)
    else:
        df = pd.read_csv(file)
    run_name = file.stem
    capacities: Dict[str, Any] = solver_results.set_index('name').loc[run_name].to_dict()
    capacities['run_name'] = run_name
    return df, capacities


def build_simulation_data(sim_name: str, 
                          participants: List[str],
                          x_variable: Dict[str, Any],
                          cost_parameters_file: Path, 
                          pre_processing_function: Callable | None = None,
                          test: bool = False) -> SimulationData:
    """
    Construct a SimulationData instance by reading all CSV files in runs_folder.
    
    Parameters:
        sim_name: Name of the simulation.
        parameters: A dictionary of simulation parameters. It may include a key 
                    "capacities" mapping run names to capacity info.

    Returns:
        A SimulationData instance.
    """
    sim_folder: Path = BASE_PATH / 'sim' / sim_name
    raw_folder: Path = sim_folder / 'raw'
    run_files: List[Path] = list(raw_folder.glob('*.csv'))

    if not pre_processing_function:
        pre_processing_function = lambda run_data: run_data

    solver_results: pd.DataFrame = pd.read_csv(sim_folder / 'results' / 'solver_results.csv')

    def wrapper_build_run_data(file: Path, solver_results: pd.DataFrame) -> RunData:
        run_data: RunData = build_run_data(file, solver_results, test=test)
        new_run_data: RunData = pre_processing_function(run_data)
        return new_run_data

    run_data: Iterator[RunData] = (wrapper_build_run_data(file, solver_results) for file in run_files)

    cost_parameters: Dict[str, Any] = load_costs(cost_parameters_file)
    
    return SimulationData(
        name=sim_name,
        run_data=run_data,
        participants=participants,
        x_variable=x_variable,
        cost_parameters=cost_parameters
    )


def compute_results(run_data_iterator: Iterator[RunData], run_analysis_function: RunAnalysisFunction) -> pd.DataFrame:
    """
    Broadcast the input run_analysis_function over all the runs in the run_data_iterator, and 
    returns the results as a DataFrame with each row corresponding to a run.
    """
    @log_execution_time
    def wrapper_analysis_function(run_data: RunData) -> Dict[str, Any]:
        _, capacities = run_data
        logger.info("Analyzing run %s", capacities['run_name'])
        return run_analysis_function(run_data)
    rows = [wrapper_analysis_function(run_data) for run_data in run_data_iterator]
    return pd.DataFrame(rows)


def default_analysis(simulation_data: SimulationData) -> pd.DataFrame:
    """
    Perform the default analysis on the simulation data.
    """
    events_config: Dict[str, Dict[str, str]] = load_events_config(simulation_data.participants)
    run_analysis_function: RunAnalysisFunction = partial(full_run_analysis, 
                                                        participants=simulation_data.participants,
                                                        costs=simulation_data.cost_parameters,
                                                        events_queries=events_config)
    return compute_results(simulation_data.run_data, run_analysis_function)


def plot_results(simulation_data: SimulationData,
                 simulation_results: pd.DataFrame):
    """
    Plots the results of the simulation as configured in the config.yaml file, using 
    the participants and x_variable attributes of the SimulationData instance.
    """
    plot_configs: Dict[str, Dict[str, Any]] = load_plot_configs(simulation_data.participants)
    x_variable: Dict[str, Any] = simulation_data.x_variable
    output_folder: Path = BASE_PATH / 'figures' / simulation_data.name

    @log_exceptions
    def plot_result(plot_config: Dict[str, Any]) -> Figure:
        return line_plot(simulation_results,
            x_variable['name'],
            plot_config['variables'],
            plot_config['title'],
            x_variable['label'],
            plot_config['y_label'])

    for plot_name, plot_config in plot_configs.items():
        fig: Figure = plot_result(plot_config)
        fig_path: Path = output_folder / f"{plot_name}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        fig.clear()


def plot_densities(simulation_data: SimulationData, overwrite: bool = False) -> None:
    """
    Plot the densities of selected variables as configured in the DENSITY_PLOTS list.
    """
    def wrapper_plot_densities(run_data: RunData):
        run_df, capacities = run_data
        run_name: str = capacities['run_name']
        output_folder: Path = BASE_PATH / 'figures' / simulation_data.name / 'runs' / run_name
        density_plots_tasks: List[Dict] = DENSITY_PLOTS
        plot_densities_run(run_df, output_folder, density_plots_tasks, overwrite=overwrite)
        return {}
    compute_results(simulation_data.run_data, wrapper_plot_densities)
    

DENSITY_PLOTS = [
    {
        "column": "marginal_cost",
    },
    {
        "column": "marginal_cost",
        "filename": "marginal_cost_less_4000",
        "condition": lambda df: df["marginal_cost"] < 4000,
    },
    {
        "column": "marginal_cost",
        "filename": "marginal_cost_positive_hydro",
        "condition": lambda df: df["hydro_marginal"] == 1,
    },
    {
        "column": "net_demand",
    },
    {
        "column": "water_level_salto",
        "x_from": 29,
    },
    {
        "column": "production_hydro"
    },
    {
        "column": "production_thermal",
        "x_to": 1500,
        "condition": lambda df: df["production_thermal"] > 1,
    }
]
