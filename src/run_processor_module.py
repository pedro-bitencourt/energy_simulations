"""
File name: run_processor_module.py

Description:
This module implements the RunProcessor class, which extends the Run class to process run results.
It extracts data such as marginal costs, price distributions, production results, and profits.

Public methods:
- RunProcessor: Initializes the RunProcessor with an existing Run instance.
- get_profits: Computes profits for the specified endogenous variables.
"""
import logging
from typing import Dict
import pandas as pd

import src.processing_module as proc
from src.participant_module import Participant
from src.run_module import Run
from src.constants import MARGINAL_COST_DF, DEMAND_DF, HOURLY_FIXED_COSTS

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WEEK_HOURS_BIN = list(range(0, 169))  # Adjusted to include 168


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
            ValueError: If the run was not successful
        """
        super().__init__(
            parent_folder=run.paths['parent_folder'],
            general_parameters=run.general_parameters,
            variables=run.variables
        )

        if not self.successful(complete=complete):
            logger.error(f'Run {self.name} was not successful.')
            raise ValueError(f'Run {self.name} was not successful.')

        self._update_paths()
        # Create dict of participants
        list_participants = ['wind', 'solar', 'thermal', 'salto']
        capacities = {var: variable['value'] for var, variable in
                      self.variables.items()}
        # WRONG AND HARDCODED, TO FIX
        capacities['salto'] = 2215

        self.participants_dict: dict[str, Participant] = {var: Participant(var,
                                                                           capacities[var],
                                                                           self.paths,
                                                                           run.general_parameters)
                                                          for var in list_participants}

    def get_random_variables_df(self, lazy=True, complete=True) -> pd.DataFrame:
        if self.paths['random_variables'].exists() and lazy:
            random_variables_df = self.load_random_variables_df()
        else:
            random_variables_df = self.construct_random_variables_df(
                complete=complete)
        return random_variables_df

    def load_random_variables_df(self) -> pd.DataFrame:
        random_variables_df = pd.read_csv(
            self.paths['random_variables'])
        return random_variables_df

    def construct_random_variables_df(self, complete=True) -> pd.DataFrame:
        def melt_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
            df = df.melt(id_vars=['datetime'],
                         var_name='scenario', value_name=name)
            return df

        if complete:
            dataframes_to_extract: dict = {'water_level': ['salto'],
                                           'production': ['wind', 'solar', 'thermal', 'salto'],
                                           'variable_costs': ['thermal'],
                                           }
        else:
            dataframes_to_extract: dict = {'water_level': [],
                                           'production': ['wind', 'solar', 'thermal'],
                                           'variable_costs': ['thermal'],
                                           }
        logging.info('Extracting random variables dataframes...')

        random_variables_df = melt_df(self.marginal_cost_df(), 'marginal_cost')

        demand_df = melt_df(self.demand_df(), 'demand')
        random_variables_df = pd.merge(random_variables_df, demand_df, on=[
                                       'datetime', 'scenario'], how='left')

        method_map = {
            'water_level': lambda p: self.water_level_participant(p),
            'production': lambda p: self.production_participant(p),
            'variable_costs': lambda p: self.variable_costs_participant(p),
            'marginal_cost': lambda _: self.marginal_cost_df()
        }

        for data_type, participants in dataframes_to_extract.items():
            method = method_map[data_type]
            for participant in participants:
                logger.info("Extracting %s for participant %s for run %s",
                            data_type, participant, self.name)
                df = method(participant)
                df = melt_df(df, f'{data_type}_{participant}')
                random_variables_df = pd.merge(random_variables_df, df, on=[
                                               'datetime', 'scenario'], how='left')

        # Process random variables
        random_variables_df = process_random_variables_df(
            random_variables_df, complete=complete)

        # Save to disk
        random_variables_df.to_csv(self.paths['random_variables'], index=False)

        return random_variables_df

    def production_participant(self, participant_key: str) -> pd.DataFrame:
        production_df: pd.DataFrame = self.participants_dict[participant_key].production_df(
        )
        return production_df

    def variable_costs_participant(self, participant_key: str) -> pd.DataFrame:
        costs_df: pd.DataFrame = self.participants_dict[participant_key].variable_costs_df(
        )
        return costs_df

    def water_level_participant(self, participant_key: str) -> pd.DataFrame:
        participant = self.participants_dict[participant_key]
        water_level_df: pd.DataFrame = participant.water_level_df(
        )
        return water_level_df

    def demand_df(self):
        demand_df = proc.open_dataframe(DEMAND_DF, self.paths['sim'])
        return demand_df

    def _update_paths(self) -> None:
        """
        Updates the paths dictionary with additional paths needed for processing.
        """
        self.paths['marginal_cost'] = self.paths['folder'] / \
            'marginal_cost.csv'
        self.paths['random_variables'] = self.paths['folder'] / \
            'random_variables.csv'
        self.paths['results_json'] = self.paths['folder'] / 'results.json'

    def marginal_cost_df(self) -> pd.DataFrame:
        """
        Extracts the marginal cost DataFrame from the simulation folder.

        Returns:
            pd.DataFrame or None: The marginal cost DataFrame, or None if extraction fails.
        """
        # Extract marginal cost DataFrame
        marginal_cost_df = proc.open_dataframe(
            MARGINAL_COST_DF, self.paths['sim'])

        # Save marginal cost DataFrame
        marginal_cost_df.to_csv(self.paths['marginal_cost'], index=False)
        logger.info(
            f'Marginal cost DataFrame saved to {self.paths["marginal_cost"]}')
        return marginal_cost_df

    def get_profits(self) -> Dict[str, float]:
        """
        Computes profits for the specified endogenous variables.

        Args:
            endogenous_variables_names (list): List of endogenous variable names.

        Returns:
            dict: A dictionary of profits.
        """
        force = self.general_parameters.get('force', False)
        lazy = not force
        # Get random variables dataframe
        random_variables_df: pd.DataFrame = self.get_random_variables_df(
            complete=False, lazy=lazy)

        random_variables_df["datetime"] = pd.to_datetime(
            random_variables_df["datetime"], errors="coerce")
        # Create a column with discount factor
        reference_data = random_variables_df['datetime'].min()
        days_diff = (random_variables_df['datetime'] - reference_data).dt.days

        random_variables_df['discount_factor'] = 1 / \
            (1 +
             self.general_parameters['annual_interest_rate'])**(days_diff / 365)

        # HARDCODED
        profits = {'wind': None, 'solar': None, 'thermal': None}
        for participant in profits.keys():
            logger.debug(
                f"Computing profits for {participant} participant.")

            capacity = self.variables[participant]['value']

            # Get present value of revenues
            revenues: float = (
                random_variables_df[f'revenues_{participant}']*random_variables_df['discount_factor']).mean()

            if participant == 'thermal':
                # Get present value of variable costs
                variable_costs: float = (
                    random_variables_df['variable_costs_thermal']*random_variables_df['discount_factor']).mean()
            else:
                variable_costs: float = 0

            # Compute average hourly variable profit for the participant
            variable_profits: float = revenues - variable_costs

            profit_per_hour_per_mw: float = variable_profits / \
                capacity - HOURLY_FIXED_COSTS[participant]

            # Normalize by the cost
            profit_normalized = profit_per_hour_per_mw / \
                (HOURLY_FIXED_COSTS[participant] + variable_costs/capacity)

            # Add profit to the dictionary
            profits[participant] = profit_normalized
        return profits

    def processed_status(self):
        if self.paths['random_variables'].exists():
            return True
        return False


def process_random_variables_df(random_variables_df, complete=True):
    def fill_daily_columns(df, variables_to_upsample):
        """
        Fills daily frequency data to match hourly frequency for specified columns.

        Parameters:
            df (pd.DataFrame): DataFrame with 'datetime' and 'scenario' columns and mixed frequency data.
            variables_to_upsample (list): List of column names that are in daily frequency.

        Returns:
            pd.DataFrame: DataFrame with hourly frequency for specified columns.
        """
        # Ensure datetime is in the correct format
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])

        result_df = df.copy()

        # Process each scenario separately
        for scenario in df['scenario'].unique():

            # Get data for this scenario
            mask = df['scenario'] == scenario
            scenario_df = df[mask].copy()

            # Set datetime as index for resampling
            scenario_df = scenario_df.set_index('datetime')
            print(scenario_df.index.is_monotonic_increasing)  # Should be True
            # Should be None or hourly ('H')
            print(scenario_df.index.freq)

            # Iterate over each column and apply the specified upsampling method
            for column, upsampling_method in variables_to_upsample.items():
                logger.debug("Upsampling %s", column)
                logger.debug(
                    f"Pre upsampling: scenario_df[{column}]={scenario_df[column]}")

                if upsampling_method == "ffill":
                    # For forward-fill:
                    # 1. Resample the column to hourly frequency.
                    # 2. Forward-fill to propagate the last valid observation forward.
                    # 3. Backward-fill to ensure even the initial periods are filled.
                    scenario_df[column] = (
                        scenario_df[column]
                        .resample('H')
                        .ffill()
                        .bfill()
                    )

                elif upsampling_method == "mean":
                    # For the "mean" method, we assume the given daily value at midnight
                    # should be spread evenly across the 24 hours.

                    # 1. Extract daily data (using 'first' to get the midnight value).
                    daily_df = scenario_df[column].resample('D').first()

                    # 2. Upsample to hourly, forward-fill the daily value to all 24 hours,
                    #    and then divide by 24 to distribute the daily total evenly.
                    hourly_df = daily_df.resample('H').ffill() / 24.0

                    logger.debug(f"Daily: daily_df[{column}]={daily_df}")
                    logger.debug(f"Hourly: hourly_df[{column}]={hourly_df}")

                    # 3. Align the hourly data back to the scenario_df's index
                    scenario_df[column] = hourly_df.reindex(
                        scenario_df.index).values

                logger.debug(
                    f"Post upsampling: scenario_df[{column}]={scenario_df[column]}")

            # Finally, write the updated results back into the original DataFrame.
            # Reindexing to ensure that the order of the timestamps matches that of result_df.
            result_df.loc[mask, variables_to_upsample.keys()] = scenario_df[variables_to_upsample.keys()].reindex(
                result_df.loc[mask, "datetime"]
            ).values

        return result_df

    if complete:
        random_variables_df['total_production'] = (random_variables_df['production_wind']
                                                   + random_variables_df['production_solar']
                                                   + random_variables_df['production_thermal']
                                                   + random_variables_df['production_salto'])
        random_variables_df['lost_load'] = random_variables_df['demand'] - \
            random_variables_df['total_production']

        variables_to_upsample = {'water_level_salto': 'ffill',
                                 'variable_costs_thermal': 'mean', }
        participants_to_revenues = ['wind', 'solar', 'thermal', 'salto']
    else:
        participants_to_revenues = ['wind', 'solar', 'thermal']
        variables_to_upsample = {'variable_costs_thermal': 'mean', }

    logger.info("Upsampling variables observed at the daily level: %s",
                variables_to_upsample)
    random_variables_df = fill_daily_columns(random_variables_df, variables_to_upsample
                                             )

    # Compute revenues
    for participant in participants_to_revenues:
        random_variables_df[f'revenues_{participant}'] = random_variables_df[
            f'production_{participant}'] * random_variables_df['marginal_cost']

    # Compute variable costs for thermal participant, HARDCODED
    random_variables_df['profits_thermal'] = random_variables_df['revenues_thermal'] - \
        random_variables_df['variable_costs_thermal']

    initial_row_count = len(random_variables_df)
    random_variables_df = random_variables_df.dropna()  # Drop rows with NaN entries
    rows_dropped = initial_row_count - len(random_variables_df)

    # Log the results
    logger.warning(
        f"Number of rows dropped due to NaN entries: {rows_dropped}")
    if rows_dropped > 3000:
        logger.critical(
            f"CRITICAL: More than 3000 rows were excluded! Total: {rows_dropped}")

    return random_variables_df
