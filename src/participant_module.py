"""
File name: participant_module.py
Author: Pedro Bitencourt
Description: This file implements the Participant class and related methods.
"""

import copy
import logging
from datetime import timedelta
from typing import Any, Dict
from pathlib import Path

import pandas as pd

from src import processing_module as proc
from src.constants import (
    COSTS,
    COSTS_BY_PARTICIPANT_TABLE,
    SCENARIOS,
    VARIABLE_COSTS_THERMAL_DF,
    SALTO_WATER_LEVEL_DF,
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
    "salto": {"folder": "HID_salto", "type": "hydro"},
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
        # Collecting parameters into a dictionary
        self.general_parameters = general_parameters
        self.type_participant = PARTICIPANTS[key_participant]["type"]

        participant_folder = PARTICIPANTS[key_participant]["folder"]
        self.dataframe_configuration = self._initialize_df_configuration(
            key_participant, paths["sim"], participant_folder
        )

    def __repr__(self):
        return f"Participant(key={self.key}, capacity={self.capacity})"

    def production_df(self) -> pd.DataFrame:
        """
        Extracts and processes the production data for the participant.
        """
        dataframe = proc.open_dataframe(self.dataframe_configuration, self.paths["sim"],
                                        daily=self.general_parameters.get("daily", False))
        logger.info(
            f"Successfully extracted and processed {self.key} production data.")
        return dataframe

    def variable_costs_df(self) -> pd.DataFrame:
        """
        Extracts and processes the variable costs data for the participant.
        """
        if self.type_participant != "thermal":
            logger.error(
                "Variable costs are only available for thermal participants.")
            raise ValueError(
                "Variable costs are only available for thermal participants.")

        dataframe = proc.open_dataframe(VARIABLE_COSTS_THERMAL_DF, self.paths["sim"],
                                        self.general_parameters.get("daily", False))
        logger.info(
            f"Successfully extracted and processed {self.key} variable costs data.")
        return dataframe

    def water_level_df(self) -> pd.DataFrame:
        """
        Extracts and processes the water level data for the participant.
        """
        if self.type_participant != "hydro":
            logger.error(
                "Water level data is only available for hydro participants.")
            raise ValueError(
                "Water level data is only available for hydro participants.")

        dataframe = proc.open_dataframe(SALTO_WATER_LEVEL_DF, self.paths["sim"],
                                        self.general_parameters.get("daily", False))
        logger.info(
            f"Successfully extracted and processed {self.key} water level data.")
        return dataframe

    def profit(self, marginal_cost_df: pd.DataFrame) -> float:
        """
        Computes the profits per MW per year for the participant,
        as a fraction of the total installation cost.
        """
        # Get average revenue per year per MW and total cost per year per MW
        average_revenue_per_year_per_mw = self._revenue(marginal_cost_df)
        total_cost_per_year_per_mw = self._cost()

        # Compute profits per year per MW
        profits_per_year_per_mw = average_revenue_per_year_per_mw - total_cost_per_year_per_mw

        # Normalize profits by total costs per year
        profits_per_year_per_mw_normalized = profits_per_year_per_mw / \
            total_cost_per_year_per_mw
        return profits_per_year_per_mw_normalized

    def _revenue(self, marginal_cost_df: pd.DataFrame) -> float:
        """
        Computes the average revenue per year per MW for the participant.
        """
        present_value_df = self._present_value_per_scenario(
            self.production_df(), marginal_cost_df)

        # Compute average revenue over scenarios
        average_total_revenue = present_value_df.mean(axis=1).values[0]

        # Access parameters from general_parameters dictionary
        years_run = self.general_parameters["years_run"]
        capacity = self.capacity

        # Compute the average revenue per year per MW
        average_total_revenue_per_year_per_mw = average_total_revenue / \
            (years_run * capacity)
        return average_total_revenue_per_year_per_mw

    def _cost(self) -> float:
        """
        Computes the total cost per year per MW for the participant.
        """
        # Compute variable cost per year
        total_variable_costs = self._total_variable_costs(
        ) if self.type_participant == "thermal" else 0
        variable_cost_per_year_per_mw = (total_variable_costs /
                                         self.general_parameters["years_run"]*self.capacity)

        # Compute fixed cost per year
        lifetime_fixed_cost_per_mw = self._lifetime_fixed_costs_per_mw()
        lifetime = COSTS[self.type_participant]["lifetime"]
        fixed_cost_per_year_per_mw = lifetime_fixed_cost_per_mw / lifetime

        # Compute total cost per year per MW
        total_cost_per_year_per_mw = variable_cost_per_year_per_mw + fixed_cost_per_year_per_mw
        return total_cost_per_year_per_mw

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
        return dataframe_configuration

    def _total_variable_costs(self) -> float:
        """
        Returns the total variable costs over the simulation period.
        """
        dataframe = proc.process_res_file(
            COSTS_BY_PARTICIPANT_TABLE, self.paths["sim"])

        if dataframe is None:
            logger.critical("Variable cost file could not be read.")
            raise FileNotFoundError

        # TO DO: Include discounting
        variable_cost = dataframe["thermal"].sum()
        return variable_cost

    def _lifetime_fixed_costs_per_mw(self) -> float:
        """
        Computes the total lifetime fixed costs.
        """

        oem_cost = COSTS[self.type_participant]["oem"]
        installation_cost = COSTS[self.type_participant]["installation"]
        lifetime = COSTS[self.type_participant]["lifetime"]
        annual_interest_rate = self.general_parameters["annual_interest_rate"]

        if annual_interest_rate > 0:
            annuity_factor = (1 - (1 + annual_interest_rate)
                              ** -lifetime) / annual_interest_rate
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

        common_index = participant_df.index.intersection(
            marginal_cost_df.index)

        if common_index.empty:
            logger.error(
                "No overlapping dates between participant and marginal cost data.")
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

    def _compute_discounted_production(self, production_df: pd.DataFrame) -> float:
        """
        Computes the discounted production for the participant.
        """
        total = 0.0
        annual_interest_rate = self.general_parameters["annual_interest_rate"]

        for scenario in SCENARIOS:
            log = scenario == 0
            total += proc.get_present_value(production_df,
                                            str(scenario), annual_interest_rate, log)
        return total / len(SCENARIOS)
