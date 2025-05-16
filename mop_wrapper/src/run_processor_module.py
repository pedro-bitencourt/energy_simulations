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
    WATER_LEVEL_DF,
    MARGINAL_COST_DF,
    DEMAND_DF
)

logger = logging.getLogger(__name__)

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

        if not self.successful():
            missing_files = Run.get_missing_files(sim_path=self.paths['sim'])
            logger.error(f'Missing files: {missing_files}')
            raise FileNotFoundError(f'Run {self.name} was not successful.')

    def construct_run_df(self, complete: bool = True) -> pd.DataFrame:
        participants: dict = self.general_parameters['participants']
        if not complete: 
            participants: dict = self.get_endogenous_participants()

        logging.info(f'Extracting dataframes for run {self.name}')

        # Non-partipant variables
        random_variables_df = self.marginal_cost_df()
        demand_df = self.demand_df()
        random_variables_df = pd.merge(random_variables_df, demand_df, on=[
                                       'datetime', 'scenario'], how='left')
        excedentes_df = get_production_df("excedentes",
                                          "IMPOEXPO_excedentes",
                                          self.paths['sim'])
        random_variables_df = pd.merge(random_variables_df, excedentes_df, on=[
                                       'datetime', 'scenario'], how='left')

        for participant in participants.keys():
            logger.debug(f'Extracting data for participant: {participant}')
            original_length = len(random_variables_df)
            # Extract production data
            df = get_production_df(participant,
                                   participants[participant]['folder'],
                                   self.paths['sim'],
                                   participants[participant]['type']
                                   )
            logger.debug("Production data for %s: %s", participant, df.head())
            random_variables_df = pd.merge(random_variables_df, df, on=[
                'datetime', 'scenario'], how='left')

            # Extract variable costs
            participant_type = participants[participant]['type']
            if participant_type in ['thermal', 'thermal_with_remainder']:
                # Extract variable costs data
                mc_thermal = self.general_parameters[
                    'cost_parameters']['marginal_cost_thermal']
                # Extract variable costs data
                df[f'variable_cost_{participant}'] = mc_thermal * df[f'production_{participant}']
                random_variables_df = pd.merge(random_variables_df, df, on=[
                    'datetime', 'scenario'], how='left')
            elif participant not in ['demand', 'excedentes']:
                random_variables_df[f'variable_cost_{participant}'] = 0

            # Extract water level data
            if participant_type == 'hydro':
                df = get_water_level_df(participant,
                                        participants[participant]['folder'],
                                        self.paths['sim'])
                random_variables_df = pd.merge(random_variables_df, df, on=[
                    'datetime', 'scenario'], how='left')

            random_variables_df = random_variables_df.dropna()
            removed_rows = original_length - len(random_variables_df)
            logger.debug(f'Removed {removed_rows} rows with NA values.')

        # Remove NA's and log the number of rows removed

        original_length = len(random_variables_df)
        random_variables_df = random_variables_df.dropna()
        removed_rows = original_length - len(random_variables_df)
        if removed_rows > 3_000:
            logger.warning(f'Over 3,000 rows were removed due to NA values.')

        logger.debug("Head of run_df: %s", random_variables_df.head())

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


def _initialize_df_configuration(key_participant: str,
                                 folder_participant: str,
                                 sim_folder: Path) -> Dict[str, Any]:
    """
    Initializes the dataframe configuration for data extraction for the given participant.
    """
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
    dataframe_configuration["filename"] = f"{sim_folder}/{folder_participant}/potencias*.xlt"
    return dataframe_configuration


def get_production_df(key_participant: str,
                      folder_participant: str,
                      sim_path: Path,
                      type_participant: str = ""
                      ) -> pd.DataFrame:
    """
    Extracts and processes the production data for the participant.
    """
    df_config = _initialize_df_configuration(key_participant,
                                             folder_participant,
                                             sim_path)
    dataframe = proc.open_dataframe(
        df_config,
        sim_path)

    dataframe = melt_df(dataframe, f"production_{key_participant}")

    if type_participant == 'thermal_with_remainder':
        df_config = _initialize_df_configuration('thermal_remainder',
                                                 'TER_thermal_remainder',
                                                 sim_path)
        remainder_dataframe = proc.open_dataframe(
            df_config,
            sim_path)
        remainder_dataframe = melt_df(
            remainder_dataframe, f"production_{key_participant}")

        dataframe[f"production_{key_participant}"] = dataframe[f"production_{key_participant}"] + \
            remainder_dataframe[f"production_{key_participant}"]

    logger.debug(
        f"Successfully extracted and processed {key_participant} production data."
    )
    return dataframe

def get_water_level_df(key_participant: str,
                       folder_participant: str, 
                       sim_path: Path) -> pd.DataFrame:
    """
    Extracts and processes the water level data for the participant.
    """
    WATER_LEVEL_DF['filename'] = f'{folder_participant}/cota*xlt'
    dataframe = proc.open_dataframe(
        WATER_LEVEL_DF,
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

def melt_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.melt(id_vars=['datetime'],
                 var_name='scenario', value_name=name)
    return df
