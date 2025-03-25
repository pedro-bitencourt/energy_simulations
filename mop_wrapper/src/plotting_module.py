from typing import List, Any, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import logging

from .utils.auxiliary import log_exceptions

logger = logging.getLogger(__name__)

AXIS_FONTSIZE = 16

def line_plot(
    df,
    x_column: str,
    y_columns: List[Dict[str, Any]],
    title: str,
    x_label: str,
    y_label: str
) -> Figure:
    """
    Generic line plot where each key in y_columns is a column in df,
    and each value is a dict containing {"label": str, "color": str}.
    
    Example of y_columns:
       [
         {name: "unconditional_production_wind", "label": "Wind (Unconditional)", "color": "#ff0000"},
         {name: "unconditional_production_solar", "label": "Solar (Unconditional)", "color": "#00ff00"}
       ]
       
    Returns:
        Matplotlib figure object.
    """
    # Order df by the x_variable
    df = df.sort_values(by=x_column)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")

    @log_exceptions
    def plot_variable(y_variable: Dict[str, Any]):
        name = y_variable['name']
        label = y_variable.get("label", name)
        color = y_variable.get("color", "#000000")  # fallback if no color
        ax.plot(
            df[x_column],
            df[name],
            marker='o',
            linestyle='-',
            label=label,
            color=color
        )
    
    for y_variable in y_columns:
        plot_variable(y_variable)
    
    ax.set_xlabel(x_label, fontsize=12)  # or use AXIS_FONTSIZE if defined
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
