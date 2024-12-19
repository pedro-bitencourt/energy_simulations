from pathlib import Path
import json
import logging


logger = logging.getLogger(__name__)


# JSON Paths
EVENTS_CONFIG_JSON_PATH: Path = Path(
    __file__).parent.parent.parent / 'config/events.json'
COMPARISONS_JSON_PATH: Path = Path(
    __file__).parent.parent.parent / 'config/comparisons.json'
PLOTS_JSON_PATH: Path = Path(
    __file__).parent.parent.parent / 'config/plots.json'
COSTS_JSON_PATH: Path = Path(
    __file__).parent.parent.parent / 'config/costs_data.json'


def load_config(config_path: Path) -> dict:
    # Load any .json configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_costs() -> dict:
    """
    Parse the costs configuration file and return a dictionary with the total fixed
    costs per hour.
    """
    costs_dict: dict[str, dict[str, float | int]
                     ] = load_config(COSTS_JSON_PATH)
    hourly_costs_dic: dict[str, float] = {
        participant: (costs_dict[participant]['installation'] / costs_dict[participant]
                      ['lifetime'] + costs_dict[participant]['oem']) / 8760
        for participant in costs_dict.keys()
    }
    return hourly_costs_dic

# Load the events configuration from the config file


def load_events() -> dict[str, str]:
    """
    Parse the events configuration file and return a dictionary with the events.
    Returns:
        dict: Dictionary with the events.
    """
    events_config = load_config(EVENTS_CONFIG_JSON_PATH)
    events = {}

    for event_name, event_config in events_config.items():
        for rhs_entry in event_config['rhs']:
            if event_config['rhs_type'] == 'value':
                event_query: str = f"{event_config['lhs']} {event_config['operator']} {rhs_entry}"
                events[f"{event_name}_{rhs_entry}"] = event_query
            elif event_config['rhs_type'] == 'percentile':
                logger.warning("Percentile not implemented yet")
                continue
    return events


def load_plots() -> dict:
    """
    Parse the plots configuration file and return a dictionary with the plots.
    Returns:
        dict: Dictionary with the plots.
    """
    return load_config(PLOTS_JSON_PATH)


def load_comparisons() -> dict:
    """
    Parse the comparisons configuration file and return a dictionary with the comparisons.
    Returns:
        dict: Dictionary with the comparisons.
    """
    return load_config(COMPARISONS_JSON_PATH)