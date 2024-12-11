import pandas as pd
import logging

from src.constants import HOURLY_FIXED_COSTS


logger = logging.getLogger(__name__)


def process_run_df(run_df: pd.DataFrame, complete=True):
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

    if complete:
        run_df['total_production'] = (run_df['production_wind']
                                      + run_df['production_solar']
                                      + run_df['production_thermal']
                                      + run_df['production_salto'])
        run_df['lost_load'] = run_df['demand'] - \
            run_df['total_production']

        variables_to_upsample = {'water_level_salto': 'ffill',
                                 'variable_costs_thermal': 'mean', }
        participants_to_revenues = ['wind', 'solar', 'thermal', 'salto']
    else:
        participants_to_revenues = ['wind', 'solar', 'thermal']
        variables_to_upsample = {'variable_costs_thermal': 'mean', }

    logger.info("Upsampling variables observed at the daily level: %s",
                variables_to_upsample)
    run_df = fill_daily_columns(run_df, variables_to_upsample
                                )

    # Compute revenues
    for participant in participants_to_revenues:
        run_df[f'revenues_{participant}'] = run_df[
            f'production_{participant}'] * run_df['marginal_cost']

    # Compute variable costs for thermal participant, HARDCODED
    run_df['profits_thermal'] = run_df['revenues_thermal'] - \
        run_df['variable_costs_thermal']

    initial_row_count = len(run_df)
    run_df = run_df.dropna()  # Drop rows with NaN entries
    rows_dropped = initial_row_count - len(run_df)

    # Log the results
    logger.warning(
        f"Number of rows dropped due to NaN entries: {rows_dropped}")
    if rows_dropped > 3000:
        logger.critical(
            f"CRITICAL: More than 3000 rows were excluded! Total: {rows_dropped}")

    return run_df


def pollution_cost_hour(run_df: pd.DataFrame, participant: str) -> float:
    if participant != 'thermal':
        return 0
    # Get present value of production
    production: float = (
        run_df[f'production_{participant}']).mean()

    # In USD per MWh
    POLLUTION_COST = 100
    pollution_cost: float = production * POLLUTION_COST
    return pollution_cost


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

    # Check if all are floats
    if not all(isinstance(x, (int, float)) for x in [revenue, variable_costs, fixed_costs, capacity_mw]):
        logger.error('One of the values is not a float')
        logger.error(
            f'{revenue=}, {variable_costs=}, {fixed_costs=}, {capacity_mw=}')
        logger.error(
            f'{type(revenue)=}, {type(variable_costs)=}, {type(fixed_costs)=}, {type(capacity_mw)=}')
        raise ValueError('One of the values is not a float')

    return {
        f'{participant}_capacity_mw': capacity_mw,
        f'{participant}_revenue_mw_hour': revenue / capacity_mw,
        f'{participant}_variable_costs_mw_hour': variable_costs / capacity_mw,
        f'{participant}_fixed_costs_mw_hour': fixed_costs,
        f'{participant}_total_costs_mw_hour': (fixed_costs + variable_costs / capacity_mw),
        f'{participant}_profits_mw_hw': (revenue - variable_costs)/capacity_mw - fixed_costs,
        f'{participant}_normalized_profits': ((revenue - variable_costs)/capacity_mw - fixed_costs) / (fixed_costs + variable_costs / capacity_mw)
    }


def conditional_means(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    # Create cutoffs dictionary
    cutoffs = {
        'water_level_salto': {
            '25': run_df['water_level_salto'].quantile(0.25),
            '10': run_df['water_level_salto'].quantile(0.10)
        },
        'production_wind': {
            '25': run_df['production_wind'].quantile(0.25),
            '10': run_df['production_wind'].quantile(0.10)
        },
        'lost_load': {
            '95': run_df['lost_load'].quantile(0.95),
            '99': run_df['lost_load'].quantile(0.99)
        },
        'profits_thermal': {
            '75': run_df['profits_thermal'].quantile(0.75),
            '95': run_df['profits_thermal'].quantile(0.95)
        }
    }

    # Add cutoffs as columns to the dataframe
    for var, percentiles in cutoffs.items():
        for perc, value in percentiles.items():
            try:
                run_df[f'{var}_cutoff_{perc}'] = value
            except KeyError:
                logger.error('Variable %s not found', var)
                continue

    queries_dict = {
        'unconditional': 'index==index',
        'water_level_34': 'water_level_salto < 34',
        'water_level_33': 'water_level_salto < 33',
        'water_level_32': 'water_level_salto < 32',
        'water_level_31': 'water_level_salto < 31',
        'drought_25': 'water_level_salto < water_level_salto_cutoff_25',
        'drought_10': 'water_level_salto < water_level_salto_cutoff_10',
        'low_wind_25': 'production_wind < production_wind_cutoff_25',
        'low_wind_10': 'production_wind < production_wind_cutoff_10',
        'drought_low_wind_25': 'water_level_salto < water_level_salto_cutoff_25 and production_wind < production_wind_cutoff_25',
        'drought_low_wind_10': 'water_level_salto < water_level_salto_cutoff_10 and production_wind < production_wind_cutoff_10',
        'blackout_95': 'lost_load > lost_load_cutoff_95',
        'blackout_99': 'lost_load > lost_load_cutoff_99',
        'negative_lost_load': 'lost_load < 0.001',
        'blackout_positive': 'lost_load > 0.001',
        'profits_thermal_75': 'profits_thermal > profits_thermal_cutoff_75',
        'profits_thermal_95': 'profits_thermal > profits_thermal_cutoff_95',
    }

    for query_name, query in queries_dict.items():
        try:
            query_frequency = run_df.query(
                query).shape[0] / run_df.shape[0]
            results_dict[f'{query_name}_frequency'] = query_frequency
            for variable in variables:
                results_dict[f'{query_name}_{variable}'] = run_df.query(query)[
                    variable].mean()
        except KeyError:
            logger.error('Query %s not successful', query_name)
            continue
    return results_dict


def intra_daily_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each hour of the day
    for hour in range(24):
        for variable in variables:
            run_df['hour'] = run_df['datetime'].dt.hour
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour']
                                                             == hour][variable].mean()
    return results_dict


def intra_weekly_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])
    # Take the mean of the variables for each hour of the week
    for hour in range(168):
        for variable in variables:
            run_df['hour_of_the_week'] = (run_df['datetime'].dt.hour +
                                          run_df['datetime'].dt.dayofweek * 24)
            results_dict[f'{variable}_hour_{hour}'] = run_df[run_df['hour_of_the_week']
                                                             == hour][variable].mean()
    return results_dict


def intra_year_averages(run_df: pd.DataFrame) -> dict:
    variables = run_df.columns.tolist()
    # Remove datetime and scenario columns
    variables.remove('datetime')
    variables.remove('scenario')

    # Initialize results dictionary
    results_dict = {}

    run_df['datetime'] = pd.to_datetime(run_df['datetime'])

    # Take the mean of the variables for each day of the year
    for day in range(365):
        for variable in variables:
            run_df['day_of_the_year'] = run_df['datetime'].dt.dayofyear
            results_dict[f'{variable}_day_{day}'] = run_df[run_df['day_of_the_year']
                                                           == day][variable].mean()

    return results_dict
