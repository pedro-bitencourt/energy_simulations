"""
File name: participant_module.py
Author: Pedro Bitencourt
Description: This file has been refactored to use pure functions instead of a Participant class.
"""
import copy
import logging
from typing import Any, Dict
from pathlib import Path
import pandas as pd

from src import processing_module as proc
from src.constants import (
    VARIABLE_COSTS_THERMAL_DF,
    SALTO_WATER_LEVEL_DF,
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


def _initialize_df_configuration(key_participant: str, sim_folder: Path) -> Dict[str, Any]:
    """
    Initializes the dataframe configuration for data extraction for the given participant.
    """
    participant_folder = PARTICIPANTS[key_participant]["folder"]
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
    dataframe_configuration["filename"] = f"{sim_folder}/{participant_folder}/potencias*.xlt"
    return dataframe_configuration

def get_production_df(key_participant: str,
                      sim_path: Path,
                      daily: bool = True) -> pd.DataFrame:
    """
    Extracts and processes the production data for the participant.
    """
    df_config = _initialize_df_configuration(key_participant, sim_path)
    dataframe = proc.open_dataframe(
        df_config,
        sim_path,
        daily=daily)
    logger.debug(
        f"Successfully extracted and processed {key_participant} production data."
    )
    return dataframe

def get_variable_costs_df(key_participant: str,
                          sim_path: Path,
                          daily: bool = True) -> pd.DataFrame:
    """
    Extracts and processes the variable costs data for the participant.
    This is only available for thermal participants.
    """
    participant_type = PARTICIPANTS[key_participant]["type"]
    if participant_type != "thermal":
        logger.error("Variable costs are only available for thermal participants.")
        raise ValueError("Variable costs are only available for thermal participants.")

    dataframe = proc.open_dataframe(
        VARIABLE_COSTS_THERMAL_DF,
        sim_path,
        daily=daily
    )
    logger.debug(
        f"Successfully extracted and processed {key_participant} variable costs data."
    )
    return dataframe


def get_water_level_df(key_participant: str,
                       sim_path: Path,
                       daily: bool = True) -> pd.DataFrame:
    """
    Extracts and processes the water level data for the participant.
    This is only available for hydro participants.
    """
    participant_type = PARTICIPANTS[key_participant]["type"]
    if participant_type != "hydro":
        logger.error("Water level data is only available for hydro participants.")
        raise ValueError("Water level data is only available for hydro participants.")

    dataframe = proc.open_dataframe(
        SALTO_WATER_LEVEL_DF,
        sim_path,
        daily=daily
    )
    logger.debug(
        f"Successfully extracted and processed {key_participant} water level data."
    )
    return dataframe
