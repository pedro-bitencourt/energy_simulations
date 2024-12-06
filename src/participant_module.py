"""
File name: participant_module.py
Author: Pedro Bitencourt
Description: This file implements the Participant class and related methods.
"""

import copy
import logging
from typing import Any, Dict

import pandas as pd

from src import processing_module as proc
from src.constants import (
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



