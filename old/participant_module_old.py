"""
Name: participant_module.py
Description:
This module contains the Participant class, which stores data from a Run participant (that is, 
a power plant or a demand participant) and some methods to process the raw output from MOP into 
variables such as the present value of revenues. 
Author: Pedro Bitencourt
"""
import os
import copy
import pandas as pd
import logging
from datetime import datetime, timedelta
from constants import SCENARIOS, POSTE_FILEPATH

from processing_module import extract_dataframe, get_present_value 


# initialize a dictionary with interest rates to be used
INTEREST_RATES = {'discounted': 0.05, 'undiscounted': 0}


PARTICIPANTS = {
    'wind': {
        'folder': 'EOLO_eoloDeci',
        'type': 'wind',
        # 'capacity_key': 'wind_capacity'
        'capacity_key': 'capacities_wind'
    },
    'solar': {
        'folder': 'FOTOV_solarDeci',
        'type': 'solar',
        # 'capacity_key': 'solar_capacity',
        'capacity_key': 'capacities_solar'
    },
    'thermal_new': {
        'folder': 'TER_thermal_new',
        'type': 'thermal',
        'capacity_key': None
    },
    'demand': {
        'folder': 'DEM_demandaPrueba',
        'type': 'demand',
        'capacity_key': None
    }
}


# OEM costs are in USD/MW/year and installation costs are in USD/MW
COSTS = {'wind': {'oem': 20_000, 'installation': 1_300_000},
         'solar': {'oem': 7_300, 'installation': 1_160_000}}
# at 5% interest rate, this gives 1_540_000 USD/MW of lifetime costs for
# wind and 1_300_000 for solar

REVENUE_STATS = [{
    'function': lambda df: df.mean(axis=1).values[0],
    'key_prefix': 'avg',
    'label_prefix': 'Average'
}]
# , {
#    'function': lambda df: df.std(axis=1).values[0],
#    'key_prefix': 'std',
#    'label_prefix': 'Standard Deviation'
# }, {
#    'function': lambda df: df.min(axis=1).values[0],
#    'key_prefix': 'min',
#    'label_prefix': 'Minimum'
# }, {
#    'function': lambda df: df.max(axis=1).values[0],
#    'key_prefix': 'max',
#    'label_prefix': 'Maximum'
# }]


class Participant:
    """
    Class to store data from a run participant and process it into final results. 
    """

    def __init__(self, key_participant, capacity, paths):
        """
        Initializes a Participant object.
        Takes the following arguments:
        key_participant: str, the key of the participant in the simulation, used to 
        extract data from the PARTICIPANTS dictionary and to name the output files.
        capacity: float, the capacity of the participant in MW.
        sim_folder: str, the folder where the simulation data is stored.
        output_folder: str, the folder where the output files will be stored.
        """
        self.key = key_participant
        participant_folder = PARTICIPANTS[key_participant]['folder']
        self.dataframe_configuration = self.initialize_df_configuration(key_participant,
                                                                        paths['sim'],
                                                                        participant_folder)
        self.output_folder = paths['output']
        self.capacity = capacity
        self.type_participant = PARTICIPANTS[key_participant]['type']
        self.paths = paths

    def __repr__(self):
        return f"Participant={self.key}, capacity={self.capacity}"

    def initialize_df_configuration(self, key_participant, sim_folder, folder_participant):
        """
        Initializes the dataframe configuration for the participant. Currently,
        only the production file is saved.
        """
        dataframe_template = {
            'folder_key': 'sim',
            'table_pattern': {
                'start': 'CANT_POSTE',
                'end': None
            },
            'columns_options': {
                'drop_columns': ['PROMEDIO_ESC'],
                'rename_columns': {
                    **{
                        f'ESC{i}': f'{i}'
                        for i in range(0, 114)
                    }, '': 'paso_start'
                },
                'numeric_columns': [f'{i}' for i in range(0, 114)]
            },
            'delete_first_row': True,
        }
        dataframe_configuration = copy.deepcopy(dataframe_template)
        # currently saving only potencias file, can be extended to other files later
        dataframe_configuration['name'] = f'{key_participant}_production'
        dataframe_configuration['filename'] = f'{sim_folder}/{folder_participant}/potencias*.xlt'
        dataframe_configuration['output_filename'] = f'{key_participant}_production'
        return dataframe_configuration

    #def add_lcoe_to_results(self, results, key_interest, discounted_production, annual_interest):
    #    lcoe_key = 'lcoe_' + self.key + "_" + key_interest
    #    oem_cost = COSTS[self.type_participant]['oem']
    #    installation_cost = COSTS[self.type_participant]['installation']
    #    results[lcoe_key] = levelized_cost_of_energy(oem_cost,
    #                                                 installation_cost,
    #                                                 discounted_production,
    #                                                 self.capacity,
    #                                                 annual_interest)
    #    return results

    def get_production_data(self, daily=False):
        production_output_path = f'{self.output_folder}/{self.key}_production_by_datetime.csv'
        # check if the production data has already been extracted
        if not os.path.exists(production_output_path):
            # extract, process, and save the production data
            extracted_dataframe = extract_dataframe(self.dataframe_configuration,
                                                         self.paths['sim'])
            if extracted_dataframe is None:
                print('Production file not found.')
                return None
            dataframe = convert_from_poste_to_datetime(
                extracted_dataframe, daily)
            dataframe.to_csv(production_output_path, index=False)
            print(
                f"Sucessfully extracted and saved {self.key} production data, with header:")
            print(dataframe.head())
        else:
            # read the production data from the saved file
            dataframe = pd.read_csv(production_output_path)
            print(f"Sucessfully read {self.key} production data from"
                  f"{production_output_path}")
        return dataframe

    def get_present_value_df(self, participant_df, marginal_cost_df, annual_interest, key_interest):
        # compute the present value of revenues for each of the 114 scenarios
        present_value_key = f'present_value_{self.key}_{key_interest}'
        present_value_df = present_value_per_scenario(participant_df,
                                                      marginal_cost_df,
                                                      annual_interest)
        return present_value_df

    def process_participant(self, daily=False):
        """
        Extracts the production data from the participants, converts it to datetime format, 
        and computes results.
        """
        results_file_path = f'{self.output_folder}/{self.key}_results.csv'

        # get the production data
        dataframe = self.get_production_data(daily)
        if dataframe is None:
            logging.error(f'Error while extracting production data for {self.key}')
            return None

        # open the marginal cost file in order to compute the revenue
        marginal_cost_df = pd.read_csv(self.paths['marginal_cost'])

        # initialize the results dictionary
        results = {}

        # iterate over the interest rates
        for key_interest, annual_interest in INTEREST_RATES.items():
            # compute the discounted expected production in MWh at the reference date
            disc_prod_key = f'production_{self.key}_{key_interest}'
            discounted_production = compute_discounted_production(
                dataframe, annual_interest)
            results[disc_prod_key] = discounted_production

            # get the present value of revenues for each scenario
            present_value_df = self.get_present_value_df(dataframe, marginal_cost_df,
                                                         annual_interest, key_interest)
            if present_value_df is None:
                logging.error(f'Error while computing present value for {self.key} at {key_interest}')
                return None

            # compute results
            results_temp = compute_results(present_value_df, discounted_production,
                                           key_interest, self.key,
                                           self.capacity)
            results.update(results_temp)

        oem_cost = COSTS[self.type_participant]['oem']
        installation_cost = COSTS[self.type_participant]['installation']
        annual_interest_rate = 0.05
        lifetime_costs_per_mw = lifetime_costs_per_mw_fun(oem_cost,
                                                          installation_cost, annual_interest_rate)
        print(f'{results=}')
        # compute profits per MW
        results[f'profits_{self.key}'] = (
            results[f'avg_revenue_{self.key}_discounted_per_mw'] - lifetime_costs_per_mw)
        # save results
        results_df = pd.DataFrame(results, index=[0])
        results_df.to_csv(results_file_path, index=False)
        return results


def compute_results(present_value_df,
                    discounted_production,
                    key_interest,
                    key_participant,
                    capacity):
    results = {}
    # iterate over different statistics of the revenues
    for stat in REVENUE_STATS:
        results_revenue_stats = revenue_stats_function(key_interest,
                                                       present_value_df,
                                                       stat,
                                                       key_participant,
                                                       capacity)
        results.update(results_revenue_stats)

    # compute discounted average price per MWh
    price_key = 'price_' + key_participant + "_" + key_interest
    results[price_key] = (results[f'avg_revenue_{key_participant}_{key_interest}']
                          / discounted_production)
    return results


def revenue_stats_function(key_interest, present_value_df, stat,
                           key_participant, capacity):
    """
    Adds a revenue statistic to the results dictionary.
    """
    results = {}
    results_key = stat['key_prefix'] + \
        "_revenue_" + key_participant + "_" + key_interest
    results[results_key] = stat['function'](present_value_df)
    if capacity is None:
        return results
    results_key_per_mw = results_key + "_per_mw"
    results[results_key_per_mw] = results[results_key] / capacity
    return results


def lifetime_costs_per_mw_fun(oem_cost: float,
                              installation_cost: float,
                              annual_interest_rate: float):
    if annual_interest_rate > 0:
        annuity_factor = (1-(1+annual_interest_rate)**-20)/annual_interest_rate
    else:
        annuity_factor = 20

    lifetime_costs_per_mw = (installation_cost + oem_cost * annuity_factor)
    return lifetime_costs_per_mw


def levelized_cost_of_energy(oem_cost: float,
                             installation_cost: float,
                             discounted_production: float,
                             capacity: float,
                             annual_interest_rate: float):
    """
    Computes the levelized cost of energy for a given participant.
    """
    lifetime_costs_per_mw: float = lifetime_costs_per_mw_fun(oem_cost,
                                                             installation_cost, annual_interest_rate)
    costs = lifetime_costs_per_mw * capacity
    return costs / discounted_production


def compute_discounted_production(production_df, annual_interest_rate):
    """
    Computes the discounted production for a given participant.
    """
    # print(production_df.head(5))
    results = pd.Series(dtype='float64')
    for scenario in SCENARIOS:
        results[scenario] = get_present_value(
            production_df, scenario, annual_interest_rate
        )
    return results.mean()


def present_value_per_scenario(participant_df, marginal_cost_df, annual_interest_rate):
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
        logging.warning(f'Warning: {num_missing_dates} dates are missing in the participant or marginal cost dataframes.')

    # Reindex the dataframes
    new_participant_df = new_participant_df.reindex(common_index)
    marginal_cost_df = marginal_cost_df.reindex(common_index)
    for scenario in SCENARIOS:
        new_participant_df['price_times_quantity'] = new_participant_df[scenario] * \
            marginal_cost_df[scenario]
        #try:
        #    new_participant_df['price_times_quantity'] = new_participant_df['price_times_quantity'].replace(
        #        r'[^\d.]', '', regex=True)
        #except AttributeError:
        #    pass
        result_scenario = get_present_value(new_participant_df,
                                                 'price_times_quantity',
                                                 annual_interest_rate)
        results_df[f'{scenario}'] = result_scenario

    revenues_df = pd.DataFrame.from_dict(results_df, orient='index').T
    return revenues_df


def convert_from_poste_to_datetime(participant_df, daily):
    """
    Converts the 'poste' column to datetime format.
    """
    if daily:
        participant_df['datetime'] = participant_df.apply(lambda row: row['paso_start'] +
                                                          timedelta(hours=row['poste']), axis=1)
        return participant_df
    return convert_from_poste_to_datetime_weekly(participant_df)


def convert_from_poste_to_datetime_weekly(participant_df):
    """
    Converts the 'poste' column to datetime format for weekly runs, using the poste
    dictionary file located at POSTE_FILEPATH.
    """
    poste_dict_df = pd.read_csv(POSTE_FILEPATH)
    scenario_columns = [col for col in poste_dict_df.columns if col.isdigit()]
    poste_dict_long = pd.melt(
        poste_dict_df,
        id_vars=['paso', 'paso_start', 'datetime'],
        value_vars=scenario_columns,
        var_name='scenario',
        value_name='poste'
    )
    #    poste_dict_long['paso'] = poste_dict_long['paso'].astype(int)
    #    poste_dict_long['poste'] = poste_dict_long['poste'].astype(int)
    #    participant_df['paso'] = participant_df['paso'].astype(int)
    #    participant_df['poste'] = participant_df['poste'].astype(int)

    poste_dict_long = poste_dict_long.astype({'paso': int, 'poste': int})
    participant_df = participant_df.astype({'paso': int, 'poste': int})
    # Convert datetime column to datetime type in poste_dict_long
    poste_dict_long['datetime'] = pd.to_datetime(poste_dict_long['datetime'],
                                                 format='%m/%d/%y %H:%M')
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
    final_result = result.pivot(
        index='datetime', columns='scenario', values='value')
    # Ensure final result is sorted by datetime
    final_result = final_result.sort_index()
    final_result['datetime'] = final_result.index
    return final_result
