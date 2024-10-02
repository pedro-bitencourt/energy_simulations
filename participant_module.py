import os
import copy
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from constants import SCENARIOS, POSTE_FILEPATH, ANNUAL_INTEREST_RATE, COSTS, DATETIME_FORMAT
#from auxiliary import cache

from processing_module import extract_dataframe, get_present_value

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

YEARS_RUN: int = 7

@dataclass
class ParticipantConfig:
    folder: str
    type: str
    capacity_key: Optional[str]


PARTICIPANTS: Dict[str, ParticipantConfig] = {
    'wind': ParticipantConfig(folder='EOLO_eoloDeci', type='wind', capacity_key='capacities_wind'),
    'solar': ParticipantConfig(folder='FOTOV_solarDeci', type='solar', capacity_key='capacities_solar'),
    'thermal_new': ParticipantConfig(folder='TER_thermal_new', type='thermal', capacity_key=None),
    'demand': ParticipantConfig(folder='DEM_demandaPrueba', type='demand', capacity_key=None)
}


REVENUE_STATS = [
    {
        'function': lambda df: df.mean(axis=1).values[0],
        'key_prefix': 'avg',
        'label_prefix': 'Average'
    }
]


class Participant:
    def __init__(self, key_participant: str, capacity: float, paths: Dict[str, str]):
        self.key = key_participant
        participant_folder = PARTICIPANTS[key_participant].folder
        self.dataframe_configuration = self._initialize_df_configuration(key_participant,
                                                                         paths['sim'],
                                                                         participant_folder)
        self.output_folder = paths['output']
        self.capacity = capacity
        self.type_participant = PARTICIPANTS[key_participant].type
        self.paths = paths

    def __repr__(self):
        return f"Participant(key={self.key}, capacity={self.capacity})"

    def _initialize_df_configuration(self, key_participant: str, sim_folder: str, folder_participant: str) -> Dict[str, Any]:
        dataframe_template = {
            'table_pattern': {
                'start': 'CANT_POSTE',
                'end': None
            },
            'columns_options': {
                'drop_columns': ['PROMEDIO_ESC'],
                'rename_columns': {
                    **{f'ESC{i}': f'{i}' for i in range(0, 114)},
                    '': 'paso_start'
                },
                'numeric_columns': [f'{i}' for i in range(0, 114)]
            },
            'delete_first_row': True,
        }
        dataframe_configuration = copy.deepcopy(dataframe_template)
        dataframe_configuration['name'] = f'{key_participant}_production'
        dataframe_configuration['filename'] = f'{sim_folder}/{folder_participant}/potencias*.xlt'
        dataframe_configuration['output_filename'] = f'{key_participant}_production'
        return dataframe_configuration

#    @cache(lambda self: f'{self.output_folder}/{self.key}_production_by_datetime.csv')
    def get_production_data(self, daily: bool = False) -> Optional[pd.DataFrame]:
        # extract the production data
        extracted_dataframe = extract_dataframe(
            self.dataframe_configuration, self.paths['sim'])

        # check if the production data was successfully extracted
        if extracted_dataframe is None:
            logging.error('Production file not found.')
            return None

        # convert the poste to datetime
        dataframe = convert_from_poste_to_datetime(
            extracted_dataframe, daily)

        logging.info(
            f"Successfully extracted and saved {self.key} production data.")
        return dataframe

    def process_participant(self, daily: bool = False) -> Optional[Dict[str, float]]:
        # extract the production data
        participant_df = self.get_production_data(daily)

        # load the marginal cost data
        marginal_cost_df = pd.read_csv(self.paths['marginal_cost'])

        # initialize the results dictionary
        results = {}

        # compute the discounted production
        disc_prod_key = f'production_{self.key}'
        discounted_production = compute_discounted_production(participant_df)
        results[disc_prod_key] = discounted_production

        # compute the present value of revenues per scenario
        present_value_df = present_value_per_scenario(
            participant_df, marginal_cost_df)
        if present_value_df is None:
            logging.error(
                f'Error while computing present value for {self.key}')
            return None

        # compute statistics of the revenues
        results_temp = compute_statistcs(
            present_value_df, discounted_production, self.key, self.capacity)

        # compute the present value of revenues per scenario
        present_value_df = present_value_per_scenario(
            participant_df, marginal_cost_df)
        if present_value_df is None:
            logging.error('Error while computing present value for %s',
                          self.key)
            return None

        # compute statistics of the revenues
        results_temp = compute_statistcs(
            present_value_df, discounted_production, self.key, self.capacity)
        results.update(results_temp)

        # compute the lifetime costs per MW
        oem_cost = COSTS[self.type_participant]['oem']
        installation_cost = COSTS[self.type_participant]['installation']
        lifetime_costs_per_mw = lifetime_costs_per_mw_fun(
            oem_cost, installation_cost)
        results[f'lifetime_costs_{self.key}_per_mw'] = lifetime_costs_per_mw

        lifetime_costs_per_mw = lifetime_costs_per_mw_fun(
            oem_cost, installation_cost)
        results[f'lifetime_costs_{self.key}_per_mw'] = lifetime_costs_per_mw

        # compute the profits
        results[f'profits_{self.key}'] = (
            results[f'avg_revenue_{self.key}_per_mw'] - lifetime_costs_per_mw)

        # free memory
        del marginal_cost_df, participant_df

        return results


def compute_statistcs(present_value_df,
                      discounted_production,
                      key_participant,
                      capacity):

    def revenue_stats_function(present_value_df, stat,
                               key_participant, capacity):
        """
        Adds a revenue statistic to the results dictionary.
        """
        results = {}
        results_key = stat['key_prefix'] + \
            "_revenue_" + key_participant
        results[results_key] = stat['function'](present_value_df)
        if capacity is None:
            return results
        results_key_per_mw = results_key + "_per_mw"
        results[results_key_per_mw] = results[results_key] / capacity
        return results

    results = {}
    # iterate over different statistics of the revenues
    for stat in REVENUE_STATS:
        results_revenue_stats = revenue_stats_function(present_value_df,
                                                       stat,
                                                       key_participant,
                                                       capacity)
        results.update(results_revenue_stats)

    # compute discounted average price per MWh
    price_key = 'price_' + key_participant
    results[price_key] = (results[f'avg_revenue_{key_participant}']
                          / discounted_production)
    return results


def lifetime_costs_per_mw_fun(oem_cost: float,
                              installation_cost: float):
    if ANNUAL_INTEREST_RATE > 0:
        annuity_factor = (1-(1+ANNUAL_INTEREST_RATE)**-YEARS_RUN)/ANNUAL_INTEREST_RATE
    else:
        annuity_factor = YEARS_RUN

    lifetime_costs_per_mw = (installation_cost + oem_cost * annuity_factor)
    return lifetime_costs_per_mw


def compute_discounted_production(production_df):
    """
    Computes the discounted production for a given participant.
    """
    # print(production_df.head(5))
    results = pd.Series(dtype='float64')
    for scenario in SCENARIOS:
        # log only for the first scenario
        log = scenario == 0
        results[scenario] = get_present_value(
            production_df, scenario, ANNUAL_INTEREST_RATE, log)
    return results.mean()


def present_value_per_scenario(participant_df, marginal_cost_df):
    """
    Computes the present value of revenues over each scenario.
    """
    results_df = {}
    new_participant_df = participant_df.copy()

    # Set the datetime column as the index
    marginal_cost_df = marginal_cost_df.set_index('datetime')
    new_participant_df = new_participant_df.set_index('datetime', drop=False)

    # ensure the 'datetime' column is parsed correctly to datetime objects
    new_participant_df.index = pd.to_datetime(new_participant_df.index)
    marginal_cost_df.index = pd.to_datetime(marginal_cost_df.index)

    # Reindex both series to a common time range and frequency
    common_index = pd.date_range(start=max(new_participant_df.index.min(), marginal_cost_df.index.min()),
                                 end=min(new_participant_df.index.max(),
                                         marginal_cost_df.index.max()),
                                 freq='H')  # Assuming hourly frequency

    # compare the common index with the index of the participant and marginal cost dataframes
    num_missing_dates = max(len(common_index) - len(new_participant_df),
                            len(common_index) - len(marginal_cost_df))
    if num_missing_dates > 0:
        logging.warning(
            f'Warning: {num_missing_dates} dates are missing in the participant or marginal cost dataframes.')

    # Reindex the dataframes
    new_participant_df = new_participant_df.reindex(common_index)
    marginal_cost_df = marginal_cost_df.reindex(common_index)
    for scenario in SCENARIOS:
        new_participant_df['price_times_quantity'] = new_participant_df[scenario] * \
            marginal_cost_df[scenario]
        result_scenario = get_present_value(new_participant_df,
                                            'price_times_quantity',
                                            ANNUAL_INTEREST_RATE)
        results_df[f'{scenario}'] = result_scenario

    revenues_df = pd.DataFrame.from_dict(results_df, orient='index').T
    return revenues_df


def convert_from_poste_to_datetime(participant_df: pd.DataFrame, daily: bool) -> pd.DataFrame:
    if daily:
        participant_df['datetime'] = participant_df.apply(
            lambda row: row['paso_start'] + timedelta(hours=row['poste']), axis=1)
        return participant_df
    return convert_from_poste_to_datetime_weekly(participant_df)


def convert_from_poste_to_datetime_weekly(participant_df: pd.DataFrame) -> pd.DataFrame:
    poste_dict_df = pd.read_csv(POSTE_FILEPATH)
    scenario_columns = [col for col in poste_dict_df.columns if col.isdigit()]
    poste_dict_long = pd.melt(
        poste_dict_df,
        id_vars=['paso', 'paso_start', 'datetime'],
        value_vars=scenario_columns,
        var_name='scenario',
        value_name='poste'
    )

    poste_dict_long = poste_dict_long.astype({'paso': int, 'poste': int})
    participant_df = participant_df.astype({'paso': int, 'poste': int})
    poste_dict_long['datetime'] = pd.to_datetime(
        poste_dict_long['datetime'], format=DATETIME_FORMAT)

    participant_long = pd.melt(
        participant_df,
        id_vars=['paso', 'poste'],
        var_name='scenario',
        value_name='value'
    )
    result = pd.merge(
        participant_long,
        poste_dict_long,
        on=['paso', 'poste', 'scenario'],
        how='left'
    )

    result = result.dropna(subset=['datetime', 'value'])
    result = result.sort_values(['datetime', 'scenario'])
    result = result.drop_duplicates(
        subset=['datetime', 'scenario'], keep='first')

    # free memory
    del participant_long, poste_dict_long

    final_result = result.pivot(
        index='datetime', columns='scenario', values='value')
    final_result = final_result.sort_index()
    final_result['datetime'] = final_result.index

    return final_result
