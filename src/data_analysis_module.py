import pandas as pd
import logging

from .constants import HOURLY_FIXED_COSTS
from .load_configs import load_events, load_variables


logger = logging.getLogger(__name__)

QUERIES_DICT: dict[str, str] = load_events()
VARIABLES: list[str] = load_variables()

PROCESSING_CONFIGS = {
    'complete': {
        'participants_to_revenues': ['wind', 'solar', 'thermal', 'salto'],
        'variables_to_upsample': {'water_level_salto': 'ffill',
                                  'variable_costs_thermal': 'mean', }
    },
    'solver': {
        'participants_to_revenues': ['wind', 'solar', 'thermal'],
        'variables_to_upsample': {'variable_costs_thermal': 'mean', }
    }
}


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

    logger.debug("Upsampling variables observed at the daily level: %s",
                 variables_to_upsample)
    logger.debug("Pre processing: %s", result_df.head())

    # Process each scenario separately
    for scenario in df['scenario'].unique():
        # Get data for this scenario
        mask = df['scenario'] == scenario
        scenario_df = df[mask].copy()

        # Set datetime as index for resampling
        scenario_df = scenario_df.set_index('datetime')
        # Iterate over each column and apply the specified upsampling method
        for column, upsampling_method in variables_to_upsample.items():
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
                # 3. Align the hourly data back to the scenario_df's index
                scenario_df[column] = hourly_df.reindex(
                    scenario_df.index).values

        result_df.loc[mask, variables_to_upsample.keys()] = scenario_df[variables_to_upsample.keys()].reindex(
            result_df.loc[mask, "datetime"]
        ).values
    result_df["datetime"] = pd.to_datetime(
        result_df["datetime"], errors="coerce")
    logger.debug("Post upsampling: %s", result_df.head())
    return result_df

def process_run_df(run_df: pd.DataFrame, complete=True):
    # Load the correct processing level
    if complete:
        procesing_configuration = PROCESSING_CONFIGS['complete']
    else:
        procesing_configuration = PROCESSING_CONFIGS['solver']

    participants_to_revenues = procesing_configuration['participants_to_revenues']
    variables_to_upsample = procesing_configuration['variables_to_upsample']

    # Upsample variables observed at the daily level to the hourly level
    logger.info("Upsampling variables observed at the daily level: %s",
                variables_to_upsample)
    run_df = fill_daily_columns(run_df, variables_to_upsample
                                )
    # Compute revenues
    for participant in participants_to_revenues:
        run_df[f'revenues_{participant}'] = run_df[
            f'production_{participant}'] * run_df['marginal_cost']

    initial_row_count = len(run_df)
    run_df = run_df.dropna()  # Drop rows with NaN entries
    rows_dropped = initial_row_count - len(run_df)

    # Log the results
    logger.debug(
        f"Number of rows dropped due to NaN entries: {rows_dropped}")
    if rows_dropped > 3000:
        logger.critical(
            f"CRITICAL: More than 3000 rows were excluded! Total: {rows_dropped}")

    return run_df


# Helper function to compute metrics for each participant
def compute_participant_metrics(run_df: pd.DataFrame, participant: str, capacity_mw: float) -> dict:
    """
    For a given run, compute the economic metrics for a given participant.

    Arguments:
        run_df (pd.DataFrame): DataFrame containing the data for the run.
        participant (str): Name of the participant.
        capacity_mw (float): Capacity of the participant in MW.
    """
    def revenue_hour(run_df: pd.DataFrame, participant: str) -> float:
        # Get present value of revenues
        revenues: float = (run_df[f'revenues_{participant}']).mean()
        return revenues

    def variable_costs_hour(run_df: pd.DataFrame, participant: str) -> float:
        if participant != 'thermal':
            return 0
        # Get present value of variable costs
        variable_costs: float = (
            run_df[f'variable_costs_{participant}']).mean()
        return variable_costs

    revenue = revenue_hour(run_df, participant)
    variable_costs = variable_costs_hour(run_df, participant)
    fixed_costs = HOURLY_FIXED_COSTS[participant]

    return {
        f'{participant}_capacity_mw': capacity_mw,
        f'{participant}_revenue_hour': revenue,
        f'{participant}_variable_costs_hour': variable_costs,
        f'{participant}_revenue_mw_hour': revenue / capacity_mw,
        f'{participant}_variable_costs_mw_hour': variable_costs / capacity_mw,
        f'{participant}_fixed_costs_mw_hour': fixed_costs,
        f'{participant}_total_costs_mw_hour': (fixed_costs + variable_costs / capacity_mw),
        f'{participant}_profits_mw_hw': (revenue - variable_costs)/capacity_mw - fixed_costs,
        f'{participant}_normalized_profits': (((revenue - variable_costs)/capacity_mw - fixed_costs) 
            / (fixed_costs + variable_costs / capacity_mw))
    }


def conditional_means(run_df: pd.DataFrame) -> dict:
    # Initialize results dictionary
    results_dict: dict = {}

    for query_name, query in QUERIES_DICT.items():
        try:
            query_frequency = run_df.query(
                query).shape[0] / run_df.shape[0]
            results_dict[f'{query_name}_frequency'] = query_frequency
            for variable in VARIABLES:
                results_dict[f'{query_name}_{variable}'] = run_df.query(query)[
                    variable].mean()
        except KeyError:
            logger.error('Query %s not successful', query_name)
            continue
    return results_dict


def intra_daily_averages(run_df: pd.DataFrame) -> dict:
    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each hour of the day
    for hour in range(24):
        for variable in VARIABLES:
            run_df['hour'] = run_df['datetime'].dt.hour
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour']
                                                             == hour][variable].mean()
    return results_dict


def intra_weekly_averages(run_df: pd.DataFrame) -> dict:
    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])
    # Take the mean of the variables for each hour of the week
    for hour in range(168):
        for variable in VARIABLES:
            run_df['hour_of_the_week'] = (run_df['datetime'].dt.hour +
                                          run_df['datetime'].dt.dayofweek * 24)
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour_of_the_week']
                                                             == hour][variable].mean()
    return results_dict


def intra_year_averages(run_df: pd.DataFrame) -> dict:
    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each day of the year
    for day in range(365):
        for variable in VARIABLES:
            run_df['day_of_the_year'] = run_df['datetime'].dt.dayofyear
            results_dict[f'{variable}_day_{day}'] = run_df[run_df['day_of_the_year']
                                                           == day][variable].mean()

    return results_dict

