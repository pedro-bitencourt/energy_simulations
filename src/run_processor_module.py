"""
File name: run_processor_module.py

Description:
    - This module implements the RunProcessor class, which extends the Run class to extract the 
    output data from MOP.

Methods:
    - RunProcessor: Initializes the RunProcessor with an existing Run instance.
    - `construct_run_df`: extracts the dataframe with all the requested data from the 
    MOP's output, as a pandas DataFrame with columns:
        - `datetime` in format
        - `scenario`: the MOP scenario for that row, from 0 to 113
        - `demand` (in MW)
        - `production_{participant}` (in MW)
        - `marginal_cost` (in $/MWh)
        - `variable_cost_{participant}` (in $)
        - `water_level_{participant}` (in m)
    

"""
import logging
from typing import Dict, Any
import copy
from pathlib import Path
import pandas as pd

from .utils import parsing_module as proc
from .run_module import Run
from .constants import (
    SALTO_WATER_LEVEL_DF,
    VARIABLE_COSTS_THERMAL_DF,
    MARGINAL_COST_DF, DEMAND_DF

)

logger = logging.getLogger(__name__)


PARTICIPANTS_LIST_ALL: list = ['wind', 'solar',
                               'thermal', 'salto', 'demand']  # , 'excedentes']
PARTICIPANTS_LIST_ENDOGENOUS: list = ['wind', 'solar', 'thermal']
PARTICIPANTS: Dict[str, Dict[str, str]] = {
    "wind": {"folder": "EOLO_eoloDeci", "type": "wind"},
    "solar": {"folder": "FOTOV_solarDeci", "type": "solar"},
    "thermal": {"folder": "TER_thermal", "type": "thermal"},
    "thermal_remainder": {"folder": "TER_thermal_remainder", "type": "thermal"},
    "demand": {"folder": "DEM_demand", "type": "demand"},
    "salto": {"folder": "HID_salto", "type": "hydro"},
    # FIX
    "excedentes": {"folder": "IMPOEXPO_excedentes", "type": "excedentes"}
}


class RunProcessor(Run):
    """
    Extends the Run class to extract data from MOP's output, such as:
        - The simulated marginal cost for each scenario, over the timeline
        - The average intraweek price distribution 
        - Total production by resource
        - Profits for each endogenous capacity

    Attributes:
        All attributes inherited from Run class
        paths (dict): Extended dictionary of relevant paths for processing
    """

    def __init__(self, run: Run, complete: bool = False):
        """
        Initializes RunProcessor with an existing Run instance.

        Args:
            run (Run): An existing Run instance to process

        Raises:
            FileNotFoundError: If the run was not successful
        """
        super().__init__(
            parent_folder=run.paths['parent_folder'],
            general_parameters=run.general_parameters,
            variables=run.variables
        )

        if not self.successful(complete=complete):
            missing_files = Run.get_missing_files(sim_path=self.paths['sim'],
                                                  complete=complete)
            logger.error(f'Run {self.name} was not successful.')
            logger.error(f'Missing files: {missing_files}')
            raise FileNotFoundError(f'Run {self.name} was not successful.')

    def construct_run_df(self, complete: bool = True) -> pd.DataFrame:
        if complete:
            participants: list = PARTICIPANTS_LIST_ALL
        else:
            participants: list = PARTICIPANTS_LIST_ENDOGENOUS

        logging.info('Extracting random variables dataframes...')

        # Extract marginal cost data
        random_variables_df = self.marginal_cost_df()
        # Extract demand data
        demand_df = self.demand_df()
        random_variables_df = pd.merge(random_variables_df, demand_df, on=[
                                       'datetime', 'scenario'], how='left')

        for participant in participants:
            logger.debug(f'Extracting data for participant: {participant}')
            original_length = len(random_variables_df)
            logger.debug(f'Length before participant %s: %s', participant,
                         original_length)
            # Extract production data
            df = get_production_df(participant, self.paths['sim'])
            logger.debug("Production data for %s: %s", participant, df.head())
            random_variables_df = pd.merge(random_variables_df, df, on=[
                'datetime', 'scenario'], how='left')

            # Extract variable costs
            participant_type = PARTICIPANTS[participant]['type']
            if participant_type == 'thermal':
                # Extract variable costs data
                df = get_variable_cost_df(participant, self.paths['sim'])
                random_variables_df = pd.merge(random_variables_df, df, on=[
                    'datetime', 'scenario'], how='left')
            elif participant not in ['demand', 'excedentes']:
                random_variables_df[f'variable_cost_{participant}'] = 0

            # Extract water level data
            if participant_type == 'hydro':
                df = get_water_level_df(participant, self.paths['sim'])
                random_variables_df = pd.merge(random_variables_df, df, on=[
                    'datetime', 'scenario'], how='left')

            random_variables_df = random_variables_df.dropna()
            removed_rows = original_length - len(random_variables_df)
            logger.debug(f'Removed {removed_rows} rows with NA values.')

        # Remove NA's and log the number of rows removed

        original_length = len(random_variables_df)
        random_variables_df = random_variables_df.dropna()
        removed_rows = original_length - len(random_variables_df)
        logger.debug(f'Removed {removed_rows} rows with NA values.')
        if removed_rows > 3_000:
            logger.warning(f'Over 3,000 rows were removed due to NA values.')

        logger.debug("Columns of run_df: %s", random_variables_df.columns)
        logger.debug("Head of run_df: %s", random_variables_df.head())

        # Save to disk
        random_variables_df.to_csv(self.paths['random_variables'], index=False)

        return random_variables_df

    def demand_df(self) -> pd.DataFrame:
        demand_df = proc.open_dataframe(DEMAND_DF, self.paths['sim'])
        demand_df = melt_df(demand_df, 'demand')
        return demand_df

    def marginal_cost_df(self) -> pd.DataFrame:
        marginal_cost_df = proc.open_dataframe(
            MARGINAL_COST_DF, self.paths['sim'])
        marginal_cost_df = melt_df(marginal_cost_df, 'marginal_cost')
        return marginal_cost_df


def _initialize_df_configuration(key_participant: str, sim_folder: Path) -> Dict[str, Any]:
    """
    Initializes the dataframe configuration for data extraction for the given participant.
    """
    participant_folder = PARTICIPANTS[key_participant]["folder"]
    dataframe_template = {
        "table_pattern": {"start": "CANT_POSTE", "end": None},
        "columns_options": {
            "drop_columns": ["PROMEDIO_ESC"],
            "rename_columns": {
                **{f"ESC{i}": f"{i}" for i in range(0, 114)},
                "": "paso_start"
            },
            "numeric_columns": [f"{i}" for i in range(0, 114)],
        },
        "delete_first_row": True,
    }
    dataframe_configuration = copy.deepcopy(dataframe_template)
    dataframe_configuration["name"] = f"{key_participant}_production"
    dataframe_configuration["filename"] = f"{sim_folder}/{participant_folder}/potencias*.xlt"
    return dataframe_configuration


def get_production_df(key_participant: str,
                      sim_path: Path
                      ) -> pd.DataFrame:
    """
    Extracts and processes the production data for the participant.
    """
    df_config = _initialize_df_configuration(key_participant, sim_path)
    dataframe = proc.open_dataframe(
        df_config,
        sim_path)

    dataframe = melt_df(dataframe, f"production_{key_participant}")

    if key_participant == 'thermal':
        df_config = _initialize_df_configuration('thermal_remainder', sim_path)
        remainder_dataframe = proc.open_dataframe(
            df_config,
            sim_path)
        remainder_dataframe = melt_df(
            remainder_dataframe, f"production_{key_participant}")

        dataframe[f"production_{key_participant}"] = dataframe[f"production_{key_participant}"] + \
            remainder_dataframe[f"production_{key_participant}"]

    # FIX ME

    logger.debug(
        f"Successfully extracted and processed {key_participant} production data."
    )
    return dataframe

# def get_variable_cost_df(key_participant: str,
#                          sim_path: Path) -> pd.DataFrame:
#    """
#    Extracts and processes the variable costs data for the participant.
#    """
#    participant_type = PARTICIPANTS[key_participant]["type"]
#    if participant_type != "thermal":
#        logger.error("Variable costs are only available for thermal participants.")
#        raise ValueError("Variable costs are only available for thermal participants.")
#
#    variable_cost_df = proc.open_dataframe(
#        variable_cost_THERMAL_DF,
#        sim_path
#    )
#    variable_cost_df = melt_df(variable_cost_df, "variable_cost")
#
#    # Get the production data
#    production_df = get_production_df(key_participant, sim_path)
#
#    # Rename the produciton column
#    production_df = production_df.rename(columns={f'production_{key_participant}': 'production'})
#
#    # Upsample the variable costs to hourly frequency
#    dataframe = upsample_scenario_proportional(variable_cost_df, production_df)
#    # Rename the variable cost column
#    dataframe = dataframe.rename(columns={'hourly_variable_cost': f'variable_cost_{key_participant}'})
#    logger.debug(
#        f"Successfully extracted and processed {key_participant} variable costs data."
#    )
#    return dataframe


def get_variable_cost_df(key_participant: str,
                         sim_path: Path) -> pd.DataFrame:
    """
    Extracts and processes the variable costs data for the participant.
    """
    participant_type = PARTICIPANTS[key_participant]["type"]
    if participant_type != "thermal":
        logger.error(
            "Variable costs are only available for thermal participants.")
        raise ValueError(
            "Variable costs are only available for thermal participants.")

    # Get the production data
    production_df = get_production_df(key_participant, sim_path)

    # Create the variable costs column
    dataframe = production_df.copy()

    # HARD CODED
    dataframe[f'variable_cost_{key_participant}'] = 192.3 * \
        dataframe[f'production_{key_participant}']

    dataframe.drop(columns=[f'production_{key_participant}'], inplace=True)

    logger.debug(
        f"Successfully extracted and processed {key_participant} variable costs data."
    )
    return dataframe


def get_water_level_df(key_participant: str,
                       sim_path: Path) -> pd.DataFrame:
    """
    Extracts and processes the water level data for the participant.
    """
    participant_type = PARTICIPANTS[key_participant]["type"]
    if participant_type != "hydro":
        logger.error(
            "Water level data is only available for hydro participants.")
        raise ValueError(
            "Water level data is only available for hydro participants.")

    dataframe = proc.open_dataframe(
        SALTO_WATER_LEVEL_DF,
        sim_path
    )
    dataframe = melt_df(dataframe, f"water_level_{key_participant}")
    dataframe = upsample_ffill(dataframe)
    logger.debug(
        f"Successfully extracted and processed {key_participant} water level data."
    )
    logger.debug("Head of water_level_df: %s", dataframe.head())
    return dataframe


def upsample_ffill(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'datetime' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Create a DataFrame to hold the result
    df_result = []

    # Iterate through each scenario
    for scenario in df['scenario'].unique():
        # Filter by the scenario
        scenario_df = df[df['scenario'] == scenario].copy()

        # Set datetime as the index for resampling
        scenario_df.set_index('datetime', inplace=True)

        # Resample to hourly and forward-fill missing values
        scenario_resampled = scenario_df.resample('h').ffill()

        # Reset the index to include datetime as a column again
        scenario_resampled.reset_index(inplace=True)

        # Append the result to the final list
        df_result.append(scenario_resampled)

    # Concatenate all the resampled scenario DataFrames
    df_result = pd.concat(df_result, ignore_index=True)

    return df_result
# def upsample_ffill(df: pd.DataFrame) -> pd.DataFrame:
#    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
#        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
#
#    df_result = df.copy()
#    logger.debug("df to upsample: %s", df.head())
#    for scenario in df['scenario'].unique():
#        scenario_mask = df['scenario'] == scenario
#        scenario_df = df[scenario_mask].copy()
#        scenario_df = scenario_df.set_index('datetime')
#
#        for column in scenario_df.columns:
#            if column in ['scenario']:
#                continue
#            scenario_df[column] = scenario_df[column].resample(
#                'h').ffill().bfill()
#        # Update in the original DataFrame
#        df_result.loc[scenario_mask, scenario_df.columns] = scenario_df.values
#
#    logger.debug("df after upsample: %s", df.head())
#    df_result.reset_index(inplace=True)
#    return df_result


def upsample_scenario_proportional(variable_cost_df: pd.DataFrame, production_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function upsamples the variable costs (which are reported on a daily frequency by MOP) to hourly frequency,
    distributing the daily costs proportionally to the hourly production.

    Args:
        variable_cost_df (pd.DataFrame): DataFrame containing the variable costs for each scenario.
            Should have columns: 'datetime','scenario', 'variable_cost'
        production_df (pd.DataFrame): DataFrame containing the production for each scenario
            Should have columns: 'datetime','scenario', 'production'
    """
    if not pd.api.types.is_datetime64_any_dtype(variable_cost_df['datetime']):
        variable_cost_df['datetime'] = pd.to_datetime(
            variable_cost_df['datetime'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(production_df['datetime']):
        production_df['datetime'] = pd.to_datetime(
            production_df['datetime'], errors='coerce')

    # Merge the two dataframes
    df = pd.merge(variable_cost_df, production_df, on=[
                  'datetime', 'scenario'], how='outer')

    results = []
    # Loop over scenarios
    for scenario in df['scenario'].unique():
        # Filter df on that scenario
        s_df = df[df['scenario'] == scenario].copy().set_index('datetime')
        s_df = s_df.reindex(pd.date_range(
            s_df.index.min(), s_df.index.max(), freq='H'))

        # Fill in the missing values for variable costs
        s_df['variable_cost'] = s_df['variable_cost'].ffill()
        s_df['production'] = s_df['production'].fillna(0)

        # Get the daily totals; if 0, replace with NA temporarily to avoid division by 0
        daily_totals = s_df['production'].groupby(
            pd.Grouper(freq='D')).transform('sum').replace(0, pd.NA)

        # Get the proportional hourly costs
        s_df['hourly_variable_cost'] = s_df['variable_cost'] * \
            (s_df['production'] / daily_totals)

        # Fill NAs with 0
        s_df['hourly_variable_cost'] = s_df['hourly_variable_cost'].fillna(0)
        s_df['scenario'] = scenario
        results.append(s_df)

    hourly_df = pd.concat(results).reset_index().rename(
        columns={'index': 'datetime'})
    # Remove the production column
    hourly_df.drop(columns='production', inplace=True)
    return hourly_df


def melt_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.melt(id_vars=['datetime'],
                 var_name='scenario', value_name=name)
    return df
