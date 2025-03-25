from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import json
import logging
import matplotlib.colors as mcolors
import colorsys




logger = logging.getLogger(__name__)


if __name__ == '__main__':
    CONFIG_PATH = '/Users/pedrobitencourt/Projects/energy_simulations/code/config/config.yaml'
else:
    from ..constants import CONFIG_PATH


def main():
    plot_configs = load_plot_configs()
    print(plot_configs)
    pass

def load_json_config(config_path: Path) -> dict:
    # Load any .json configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

### Auxiliary functions
def load_costs(costs_path) -> dict[str, dict]:
    #    if costs_path is None:
    #        costs_path = to do: set a default
    costs_dict: dict[str, dict[str, float | int]
                     ] = load_json_config(costs_path)
    hourly_costs_dic: dict[str, float] = {
        f"fixed_cost_{participant}": (costs_dict[participant]['installation'] / costs_dict[participant]
                      ['lifetime'] + costs_dict[participant]['oem']) / 8760
        for participant in costs_dict.keys()
    }

    marginal_costs_dict: dict[str, float] = {
        f"marginal_cost_{participant}": costs_dict[participant].get('marginal_cost', 0)
        for participant in costs_dict.keys()
    }

    result: dict[str, float] = {
        **hourly_costs_dic,
        **marginal_costs_dict
    }
    result['marginal_cost_hydro'] = 0
    result['fixed_cost_hydro'] = 0
    return result


def load_yaml_config()-> Dict[str, Any]:
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_plot_configs() -> Dict[str, Dict[str, Any]]:
    config = load_yaml_config()
    comparisons_dict: Dict[str, list[Dict[str, Any]]] = config['comparisons']
    events_dict: Dict[str, Dict[str, Any]] = config['events']
    plots_config: Dict[str, Any] = config['plots']
    # Assume there is a separate section for comparison plots
    comparison_plots_raw: Dict[str, Any] = plots_config['events']

    def create_comparison_plot_config(
        comparison_item: tuple[str, list[Dict[str, Any]]],
        plot_item: tuple[str, Dict[str, Any]]
    ) -> Dict:
        comparison_name, comparison_events_list = comparison_item
        plot_name, plot_config = plot_item
        y_columns = [
            {   "name": f"mean_{y_var['name']}_{event}",
                "label": f"{y_var['label']} ({events_dict[event]['label']})",
                "color": get_color(comparison_events_list, event, y_var)
            }
            for event in comparison_events_list
            for y_var in plot_config['variables']
        ]
        return {
            f"comparisons/{comparison_name}/{plot_name}": {
                "variables": y_columns,
                "title": plot_config['title'],
                "y_label": plot_config['y_label']
            }
        }

    comparison_plots: Dict[str, Any] = {}
    for comparison_item in comparisons_dict.items():
        for plot_item in comparison_plots_raw.items():
            new_config: Dict[str, Dict[str, Any]] = create_comparison_plot_config(
                comparison_item, plot_item)
            comparison_plots.update(new_config)
    new_plot_configs = {**plots_config['general'], **comparison_plots}
    return new_plot_configs



def adjust_lightness(color, amount=1.0):
    """
    Lighten or darken the given color by multiplying its lightness (in HLS space) by 'amount'.
    amount < 1 => darker
    amount > 1 => lighter
    """
    # Parse the color to RGB
    c = mcolors.to_rgb(color)  # Normalize to (R, G, B) in [0, 1]
    h, l, s = colorsys.rgb_to_hls(*c)
    l = max(0, min(1, l * amount))  # Clamp between 0 and 1
    return colorsys.hls_to_rgb(h, l, s)

def get_color(comparison_config: List[Dict], event: str, y_variable: Dict):
    #  Compute a per-(event, variable) color *here*
    base_color = y_variable.get("color", "#ffd9b3")  # fallback
    event_index = comparison_config.index(event)
    factor = 1 - 0.35 * event_index # You can tweak the factor to adjust the color
    line_color = adjust_lightness(base_color, factor)
    return line_color

if __name__ == '__main__':
    main()
