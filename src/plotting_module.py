from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from .utils.load_configs import load_events_labels
import logging
import colorsys

logger = logging.getLogger(__name__)

AXIS_FONTSIZE = 16


def adjust_lightness(color, amount=1.0):
    """
    Lighten or darken the given color by multiplying its lightness (in HLS space) by 'amount'.
    amount < 1 => darker
    amount > 1 => lighter
    """
    # Parse the color to RGB
    c = mcolors.to_rgb(color)  # Normalize to (R, G, B) in [0, 1]
    
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*c)
    
    # Adjust the lightness
    l = max(0, min(1, l * amount))  # Clamp between 0 and 1
    
    # Convert back to RGB
    return colorsys.hls_to_rgb(h, l, s)

def line_plot(
    df,
    x_column: str,
    # y_columns is now { combined_col_name: {"label": "...", "color": "..."} }
    y_columns: dict[str, dict],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path
):
    """
    Generic line plot where each key in y_columns is a column in df,
    and each value is a dict containing {"label": str, "color": str}.
    
    Example of y_columns:
       {
         "unconditional_production_wind": {"label": "Wind (Unconditional)", "color": "#ff0000"},
         "unconditional_production_solar": {"label": "Solar (Unconditional)", "color": "#00ff00"}
       }
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    for col_name, info in y_columns.items():
        if col_name not in df.columns:
            continue
        
        label = info.get("label", col_name)
        color = info.get("color", "#000000")  # fallback if no color
        
        plt.plot(
            df[x_column],
            df[col_name],
            marker='o',
            linestyle='-',
            label=label,
            color=color
        )

    plt.xlabel(x_label, fontsize=AXIS_FONTSIZE)
    plt.ylabel(y_label, fontsize=AXIS_FONTSIZE)
    plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.savefig(f"{output_path}_no_title", dpi=300)
#    plt.title(title)
    plt.savefig(output_path, dpi=300)
    plt.close()

def event_comparison_plot(
    events_data: dict[str, pd.DataFrame],
    events_config: list,  # or list of events
    y_variables: list[dict],
    y_variable_axis: str,
    x_variable: dict,
    file_path: Path,
    title: str
):
    """
    Create a comparison line plot for multiple events and variables.

    Args:
        events_data (dict[str, pd.DataFrame]): Data for each event.
        events_config (dict or list): Contains or enumerates the event names we want to plot.
        y_variables (list[dict]): Each dict has {"name": str, "label": str, "color": optional str}.
        y_variable_axis (str): Label for the Y-axis.
        x_variable (dict): {"name": str, "label": str} for the X-axis.
        file_path (Path): Where to save the final plot.
        title (str): Plot title.
    """
    logger.info(
        "Starting event_comparison_plot with %s, %s, %s, %s, %s, %s",
        events_config, y_variables, y_variable_axis, x_variable, file_path, title
    )

    # Load event labels for nice display
    events_labels: dict[str, str] = load_events_labels()

    # Convert events_config into a list if it's not already
    events_list = list(events_config) if isinstance(events_config, dict) else events_config

    x_col = x_variable["name"]
    combined_df = pd.DataFrame()

    # This will map: combined_col_name -> {"label": "...", "color": "..."}
    y_columns = {}

    # -----------------------------
    #  Build the combined DataFrame
    # -----------------------------
    for event_name in events_list:
        if event_name not in events_data:
            logger.warning("Event '%s' not found in events_data. Skipping.", event_name)
            continue

        event_df = events_data[event_name].copy()

        if x_col not in event_df.columns:
            logger.warning(
                "X column '%s' not found in data for event '%s'. Skipping.",
                x_col, event_name
            )
            continue

        # Sort by x_column for consistent lines
        event_df = event_df.sort_values(by=x_col)

        if combined_df.empty:
            combined_df[x_col] = event_df[x_col].values

        # For each requested y-variable
        for var_cfg in y_variables:
            var_name = var_cfg["name"]
            var_label = var_cfg["label"]
            base_color = var_cfg.get("color", "#ffd9b3")  # fallback

            if var_name not in event_df.columns:
                logger.warning(
                    "Variable '%s' not found for event '%s'. Skipping.",
                    var_name, event_name
                )
                continue

            # Build the new column name, e.g. "unconditional_production_wind"
            combined_col_name = f"{event_name}_{var_name}"
            combined_df[combined_col_name] = event_df[var_name].values

            # "Pretty" label that includes event info
            # We look up the event label in events_labels (fallback to event_name if missing)
            event_label = events_labels.get(event_name, event_name)
            if event_label:
                full_label = f"{var_label} ({event_label})"
            else:
                full_label = var_label
            
            # ------------------------------------------------
            #  Compute a per-(event, variable) color *here* 
            # ------------------------------------------------
            try:
                # We'll lighten/darken the base color depending on event_index
                event_index = events_list.index(event_name)
            except ValueError:
                event_index = 0

            # E.g. if you have multiple events, you can tweak the factor by event index
            factor = 1 - 0.35 * event_index
            line_color = adjust_lightness(base_color, factor)

            # Store label + color
            y_columns[combined_col_name] = {
                "label": full_label,
                "color": line_color
            }

    # ----------------------------------------------------
    #  Call the generic line_plot with our color-infused y_columns
    # ----------------------------------------------------
    line_plot(
        df=combined_df,
        x_column=x_col,
        y_columns=y_columns,  # {combined_col_name: {"label":..., "color":...}}
        title=title,
        x_label=x_variable["label"],
        y_label=y_variable_axis,
        output_path=file_path
    )

    logger.info("Plot saved to: %s", file_path)
