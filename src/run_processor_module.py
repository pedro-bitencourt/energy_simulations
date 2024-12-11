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
import json

import src.processing_module as proc
from src.participant_module import get_production_df, get_variable_costs_df, get_water_level_df
from src.run_module import Run
from src.constants import MARGINAL_COST_DF, DEMAND_DF
from src.data_analysis_module import process_run_df, compute_participant_metrics

logger = logging.getLogger(__name__)

PARTICIPANTS_COMPLETE: list = ['wind', 'solar', 'thermal', 'salto']
PARTICIPANTS: list = ['wind', 'solar', 'thermal']


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

            logger.error(f'Run {self.name} was not successful.')
            raise FileNotFoundError(f'Run {self.name} was not successful.')

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
            'water_level': lambda p: get_water_level_df(p, self.paths['sim']),
            'production': lambda p: get_production_df(p, self.paths['sim']),
            'variable_costs': lambda p: get_variable_costs_df(p, self.paths['sim']),
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
        random_variables_df = process_run_df(
            random_variables_df, complete=complete)

        # Save to disk
        random_variables_df.to_csv(self.paths['random_variables'], index=False)

        return random_variables_df

    def demand_df(self):
        demand_df = proc.open_dataframe(DEMAND_DF, self.paths['sim'])
        return demand_df

    def marginal_cost_df(self) -> pd.DataFrame:
        """
        Extracts the marginal cost DataFrame from the simulation folder.

        Returns:
            pd.DataFrame or None: The marginal cost DataFrame, or None if extraction fails.
        """
        # Extract marginal cost DataFrame
        marginal_cost_df = proc.open_dataframe(
            MARGINAL_COST_DF, self.paths['sim'])
        return marginal_cost_df

    def profits_data_dict(self, complete: bool = False) -> Dict:
        """
        Computes profits for the specified endogenous variables.

        Returns:
            dict: A dictionary of profits.
        """
        # Get random variables dataframe
        run_df: pd.DataFrame = self.construct_random_variables_df(
            complete=False)

        if complete:
            participants: list = PARTICIPANTS_COMPLETE
        else:
            participants: list = PARTICIPANTS

        results_dict: dict = {
            f'{participant}_capacity': self.variables[participant]['value']
            for participant in participants
        }

        # Update the results dictionary with metrics for each participant
        for participant in participants:
            capacity: float = self.variables[participant]['value']
            results_dict.update(compute_participant_metrics(
                run_df, participant, capacity))

        if complete:
            system_total_cost: float = sum(
                [results_dict[f'{participant}_total_cost'] for participant in PARTICIPANTS_COMPLETE])
            system_fixed_cost: float = sum(
                [results_dict[f'{participant}_fixed_cost'] for participant in PARTICIPANTS_COMPLETE])
            results_dict.update({
                'system_total_cost': system_total_cost,
                'system_fixed_cost': system_fixed_cost,
            })

        return results_dict

    def get_profits(self):
        """
        Computes profits for the specified endogenous variables.

        Returns:
            dict: A dictionary of profits.
        """
        profits_data: dict = self.profits_data_dict(complete=False)
        json_path = self.paths['folder'] / 'profits_data.json'
        with open(json_path, "w") as f:
            json.dump(profits_data, f)

        logger.debug("profits_data for %s:", self.name)
        print(profits_data)
        profits_dict: dict = {participant: profits_data[f'{participant}_normalized_profits']
                              for participant in PARTICIPANTS}
        return profits_dict
