from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

import logging

logger = logging.getLogger(__name__)

def generate_colot_palette(total_lines: int):

    # Create a colorblind-friendly color palette
    # Using a combination of IBM ColorBlind Safe palette and Color Brewer
    base_colors = [
        '#648FFF',  # Blue
        '#DC267F',  # Magenta
        '#785EF0',  # Purple
        '#FE6100',  # Orange
        '#FFB000',  # Gold
        '#009E73',  # Green
        '#56B4E9',  # Light Blue
        '#E69F00',  # Brown
        '#CC79A7',  # Pink
        '#000000'   # Black
    ]
    
    
    # If we need more colors than in base_colors, create variations
    if total_lines > len(base_colors):
        # Create variations by adjusting lightness
        colors = []
        for color in base_colors:
            rgb = mcolors.to_rgb(color)
            hsv = mcolors.rgb_to_hsv(rgb)
            # Create a darker and lighter version of each color
            colors.extend([
                mcolors.hsv_to_rgb((hsv[0], hsv[1], min(1, hsv[2] * 0.7))),  # Darker
                rgb,
                mcolors.hsv_to_rgb((hsv[0], hsv[1], min(1, hsv[2] * 1.3)))   # Lighter
            ])
    else:
        colors = base_colors

    return colors

def line_plot(
    df, 
    x_column: str, 
    y_columns: dict[str, str],  # {y_col_name: y_label}
    title: str, 
    x_label: str,
    y_label: str, 
    output_path: Path
):
    """
    Create a line plot from a single DataFrame. Assumes df is already clean and sorted.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame containing all data to be plotted.
        x_column (str): The name of the column to use as the x-axis.
        y_columns (dict[str,str]): A dictionary mapping from the column name to a display label.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        output_path (Path): File path to save the plot image.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    for col, label in y_columns.items():
        if col not in df.columns:
            continue
        plt.plot(
            df[x_column],
            df[col],
            marker='o',
            linestyle='-',
            label=label
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Variables')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # close figure to free memory

def event_comparison_plot(events_data: dict[str, pd.DataFrame],
                     events_config: dict,
                     y_variables: list[dict],
                     y_variable_axis: str,
                     x_variable: dict,
                     file_path: Path,
                     title: str):
    """
    Create a comparison line plot for multiple events and variables, using the generic line_plot function.
    
    Args:
        events_data (dict[str, pd.DataFrame]): A dictionary containing data for each event.
        events_config (dict): A dictionary containing event configuration, including the 'events' key.
        y_variables (list[dict]): A list of dictionaries containing the name and label of each variable.
        y_variable_axis (str): Label for the Y-axis.
        x_variable (dict): A dictionary containing the name and label of the x variable.
        file_path (Path): File path to save the resulting plot.
        title (str): Title of the plot.
    """

    logger.info("Starting the event_comparison function with %s, %s, %s, %s, %s, %s",
                events_config, y_variables, y_variable_axis, x_variable, file_path, title)

    # Extract list of events from config
    events = events_config.get('events', [])
    if not events:
        logger.warning("No events provided in events_config.")
        return

    # We'll create a combined DataFrame that includes all events and variables.
    # This DataFrame will have:
    #   - One x_column (e.g., 'lake_factor')
    #   - Multiple y_columns, one for each event-variable pair.

    x_col = x_variable['name']
    combined_df = pd.DataFrame()

    # Dictionary that maps combined column names to their display labels
    y_columns = {}

    for event in events:
        # Check if this event data is available
        if event not in events_data:
            logger.warning("Event '%s' not found in events_data. Skipping.", event)
            continue

        event_df = events_data[event].copy()

        # Check if x_column is present
        if x_col not in event_df.columns:
            logger.warning("x_column '%s' not found in event '%s' DataFrame. Skipping event.", x_col, event)
            continue

        # Sort by x-column
        event_df = event_df.sort_values(by=x_col)

        # Initialize combined_df if empty
        if combined_df.empty:
            combined_df[x_col] = event_df[x_col].values

        # For each y-variable, add a column for this event
        for y_var in y_variables:
            var_name = y_var['name']
            var_label = y_var['label']

            if var_name not in event_df.columns:
                logger.warning("Variable '%s' not found in event '%s' DataFrame. Skipping variable.",
                               var_name, event)
                continue

            # Create a combined column name: e.g. "unconditional_production_thermal"
            combined_col_name = f"{event}_{var_name}"

            # Add the data to the combined_df
            combined_df[combined_col_name] = event_df[var_name].values

            # Create a label that includes the event and variable label: "Event, Variable Label"
            y_columns[combined_col_name] = f"{event}, {var_label}"

    # After we've processed all events and variables, we have:
    # combined_df: A DataFrame with one x_col and multiple y-columns.
    # y_columns: A dict mapping each y_column to a nicely formatted label.

    # If combined_df is empty or no y_columns were added, nothing to plot.
    if combined_df.empty or not y_columns:
        logger.warning("No data available for plotting after processing events and variables.")
        return

    # Now we can call the generic line_plot function.
    line_plot(
        df=combined_df,
        x_column=x_col,
        y_columns=y_columns,
        title=title,
        x_label=x_variable['label'],
        y_label=y_variable_axis,
        output_path=file_path
    )
    logger.info("Plot saved to: %s", file_path)
