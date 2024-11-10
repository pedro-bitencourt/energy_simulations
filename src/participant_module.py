"""
File name: participant_module.py
Author: Pedro Bitencourt
Description: This file implements the Participant class and related methods.
"""

import copy
import logging
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src import processing_module as proc
from src.constants import (
    COSTS,
    COSTS_BY_PARTICIPANT_TABLE,
    DATETIME_FORMAT,
    POSTE_FILEPATH,
    SCENARIOS,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Participant configuration mapping
PARTICIPANTS: Dict[str, Dict[str, str]] = {
    "wind": {"folder": "EOLO_eoloDeci", "type": "wind"},
    "solar": {"folder": "FOTOV_solarDeci", "type": "solar"},
    "thermal": {"folder": "TER_new_thermal", "type": "thermal"},
    "demand": {"folder": "DEM_demandaPrueba", "type": "demand"},
}


class Participant:
    """
    Represents a participant in the energy market, such as a wind farm,
    solar plant, or thermal plant.
    """

    def __init__(
        self,
        key_participant: str,
        capacity: float,
        paths: Dict[str, str],
        general_parameters: Dict[str, Any],
    ):
        self.key = key_participant
        self.capacity = capacity
        self.paths = paths
        self.general_parameters = general_parameters  # Collecting parameters into a dictionary
        self.type_participant = PARTICIPANTS[key_participant]["type"]

        participant_folder = PARTICIPANTS[key_participant]["folder"]
        self.dataframe_configuration = self._initialize_df_configuration(
            key_participant, paths["sim"], participant_folder
        )

    def __repr__(self):
        return f"Participant(key={self.key}, capacity={self.capacity})"

    def profit(self, marginal_cost_df: pd.DataFrame) -> float:
        """
        Computes the profits per MW per year for the participant,
        as a fraction of the total installation cost.
        """
        average_revenue_per_year_per_mw = self._revenue(marginal_cost_df)
        total_cost_per_year_per_mw = self._cost()
        profits_per_year_per_mw = average_revenue_per_year_per_mw - total_cost_per_year_per_mw

        installation_cost = COSTS[self.type_participant]["installation"]
        profits = profits_per_year_per_mw / installation_cost
        return profits

    def _revenue(self, marginal_cost_df: pd.DataFrame) -> float:
        """
        Computes the average revenue per year per MW for the participant.
        """
        participant_df = self._get_production_data()

        present_value_df = self._present_value_per_scenario(participant_df, marginal_cost_df)

        # Compute average revenue over scenarios
        average_revenue = present_value_df.mean(axis=1).values[0]

        # Access parameters from general_parameters dictionary
        years_run = self.general_parameters["years_run"]
        capacity = self.capacity

        # Compute the average revenue per year per MW
        average_revenue_per_year_per_mw = average_revenue / (years_run * capacity)
        return average_revenue_per_year_per_mw

    def _cost(self) -> float:
        """
        Computes the average total cost per year per MW for the participant.
        """
        oem_cost = COSTS[self.type_participant]["oem"]
        installation_cost = COSTS[self.type_participant]["installation"]
        lifetime = COSTS[self.type_participant]["lifetime"]

        variable_costs = self._get_variable_costs() if self.type_participant == "thermal" else 0

        lifetime_fixed_cost = self._lifetime_fixed_costs(oem_cost, installation_cost, lifetime)
        fixed_cost_per_year = lifetime_fixed_cost / lifetime

        # Access parameters from general_parameters dictionary
        years_run = self.general_parameters["years_run"]
        capacity = self.capacity

        variable_cost_per_year = variable_costs / years_run

        total_cost_per_year_per_mw = (fixed_cost_per_year + variable_cost_per_year) / capacity
        return total_cost_per_year_per_mw

    def _get_production_data(self) -> pd.DataFrame:
        """
        Extracts and processes the production data for the participant.
        """
        extracted_dataframe = proc.extract_dataframe(self.dataframe_configuration, self.paths["sim"])

        if extracted_dataframe is None:
            logger.critical("Production file not found.")
            raise FileNotFoundError

        daily = self.general_parameters.get("daily", False)  # Access 'daily' from general_parameters
        dataframe = self._convert_from_poste_to_datetime(extracted_dataframe, daily)
        logger.info(f"Successfully extracted and processed {self.key} production data.")
        return dataframe

    def _initialize_df_configuration(
        self, key_participant: str, sim_folder: str, folder_participant: str
    ) -> Dict[str, Any]:
        """
        Initializes the dataframe configuration for data extraction.
        """
        dataframe_template = {
            "table_pattern": {"start": "CANT_POSTE", "end": None},
            "columns_options": {
                "drop_columns": ["PROMEDIO_ESC"],
                "rename_columns": {**{f"ESC{i}": f"{i}" for i in range(0, 114)}, "": "paso_start"},
                "numeric_columns": [f"{i}" for i in range(0, 114)],
            },
            "delete_first_row": True,
        }
        dataframe_configuration = copy.deepcopy(dataframe_template)
        dataframe_configuration["name"] = f"{key_participant}_production"
        dataframe_configuration["filename"] = f"{sim_folder}/{folder_participant}/potencias*.xlt"
        dataframe_configuration["output_filename"] = f"{key_participant}_production"
        return dataframe_configuration

    def _get_variable_costs(self) -> float:
        """
        Returns the variable costs of the thermal participant.
        """
        dataframe = proc.process_res_file(COSTS_BY_PARTICIPANT_TABLE, self.paths["sim"])

        if dataframe is None:
            logger.critical("Variable cost file could not be read.")
            raise FileNotFoundError

        variable_cost = dataframe["thermal"].sum()
        return variable_cost

    def _lifetime_fixed_costs(self, oem_cost: float, installation_cost: float, lifetime: int) -> float:
        """
        Computes the total lifetime fixed costs.
        """
        annual_interest_rate = self.general_parameters["annual_interest_rate"]

        if annual_interest_rate > 0:
            annuity_factor = (1 - (1 + annual_interest_rate) ** -lifetime) / annual_interest_rate
        else:
            annuity_factor = lifetime

        lifetime_costs = installation_cost + oem_cost * annuity_factor
        return lifetime_costs

    def _present_value_per_scenario(
        self, participant_df: pd.DataFrame, marginal_cost_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes the present value of revenues over each scenario.
        """
        results_df = {}
        participant_df = participant_df.copy()
        marginal_cost_df = marginal_cost_df.copy()

        participant_df.set_index("datetime", inplace=True)
        
        logger.debug(f"Marginal cost df: {marginal_cost_df.head()}")
        marginal_cost_df.set_index("datetime", inplace=True)

        participant_df.index = pd.to_datetime(participant_df.index)
        marginal_cost_df.index = pd.to_datetime(marginal_cost_df.index)

        common_index = participant_df.index.intersection(marginal_cost_df.index)

        if common_index.empty:
            logger.error("No overlapping dates between participant and marginal cost data.")
            raise ValueError("No overlapping dates for computation.")

        participant_df = participant_df.reindex(common_index)
        marginal_cost_df = marginal_cost_df.reindex(common_index)

        annual_interest_rate = self.general_parameters["annual_interest_rate"]

        for scenario in SCENARIOS:
            scenario_str = str(scenario)
            participant_df["price_times_quantity"] = (
                participant_df[scenario_str] * marginal_cost_df[scenario_str]
            )
            # Reset index to get datetime as a column
            temp_df = participant_df.reset_index()
            result_scenario = proc.get_present_value(
                temp_df, "price_times_quantity", annual_interest_rate
            )
            results_df[scenario_str] = result_scenario

        revenues_df = pd.DataFrame.from_dict(results_df, orient="index").T
        return revenues_df

    def _convert_from_poste_to_datetime(self, participant_df: pd.DataFrame, daily: bool) -> pd.DataFrame:
        """
        Converts 'poste' time format to datetime.
        """
        if daily:
            if participant_df.columns[-1] == "paso_start" and "paso_start" in participant_df.columns[:-1]:
                participant_df = participant_df.iloc[:, :-1]

            participant_df["paso_start"] = pd.to_datetime(
                participant_df["paso_start"], format="%Y/%m/%d/%H:%M:%S"
            )
            participant_df["datetime"] = participant_df.apply(
                lambda row: row["paso_start"] + timedelta(hours=float(row["poste"])), axis=1
            )
            return participant_df
        else:
            return self._convert_from_poste_to_datetime_weekly(participant_df)

    def _convert_from_poste_to_datetime_weekly(self, participant_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts 'poste' time format to datetime for weekly data.
        """
        poste_dict_df = pd.read_csv(POSTE_FILEPATH)
        scenario_columns = [col for col in poste_dict_df.columns if col.isdigit()]
        poste_dict_long = pd.melt(
            poste_dict_df,
            id_vars=["paso", "paso_start", "datetime"],
            value_vars=scenario_columns,
            var_name="scenario",
            value_name="poste",
        )

        poste_dict_long = poste_dict_long.astype({"paso": int, "poste": int})
        participant_df = participant_df.astype({"paso": int, "poste": int})
        poste_dict_long["datetime"] = pd.to_datetime(
            poste_dict_long["datetime"], format=DATETIME_FORMAT
        )

        participant_long = pd.melt(
            participant_df, id_vars=["paso", "poste"], var_name="scenario", value_name="value"
        )

        result = pd.merge(
            participant_long,
            poste_dict_long,
            on=["paso", "poste", "scenario"],
            how="left",
        )

        result = result.dropna(subset=["datetime", "value"])
        result = result.sort_values(["datetime", "scenario"])
        result = result.drop_duplicates(subset=["datetime", "scenario"], keep="first")

        final_result = result.pivot(index="datetime", columns="scenario", values="value")
        final_result = final_result.sort_index()
        final_result["datetime"] = final_result.index

        return final_result

    def _compute_discounted_production(self, production_df: pd.DataFrame) -> float:
        """
        Computes the discounted production for the participant.
        """
        total = 0.0
        annual_interest_rate = self.general_parameters["annual_interest_rate"]

        for scenario in SCENARIOS:
            log = scenario == 0
            total += proc.get_present_value(production_df, str(scenario), annual_interest_rate, log)
        return total / len(SCENARIOS)
