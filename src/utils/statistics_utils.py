import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np 
from statsmodels.nonparametric.kernel_density import KDEMultivariate 
from pygam import LogisticGAM, s 
import matplotlib.pyplot as plt 

from statsmodels.nonparametric.kde import KDEUnivariate


from .auxiliary import skip_if_exists

logger = logging.getLogger(__name__)


def compute_kde(series: pd.Series, x_from=None, x_to=None, bandwidth=1.0):
    kde = KDEUnivariate(series)
    kde.fit(kernel="gau", bw=bandwidth)  # <-- only in KDEUnivariate

    x_min = series.min() if x_from is None else x_from
    x_max = series.max() if x_to is None else x_to
    x_vals = np.linspace(x_min, x_max, 200)
    density = kde.evaluate(x_vals)       # <-- also only in KDEUnivariate

    return x_vals, density


def plot_density(
    series: pd.Series,
    variable_name: str,
    out_path: Path,
    x_from: Optional[float] = None,
    x_to: Optional[float] = None,
    bandwidth: float = 1.0,
    overwrite: bool = False
) -> None:

    if skip_if_exists(out_path, overwrite):
        return

    title = f"Density Plot: {variable_name.replace('_', ' ').title()}"

    sampled_series = series.sample(min(250_000, len(series)))
    x_vals, density = compute_kde(sampled_series, x_from, x_to, bandwidth)

    plt.figure()
    plt.plot(x_vals, density, label="Density")
    plt.title(title)
    plt.xlabel(variable_name.replace("_", " ").title())
    plt.ylabel("Density")

    if x_to is not None:
        plt.xlim(x_from if x_from is not None else x_vals.min(),
                 x_to if x_to is not None else x_vals.max())

    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved density plot to %s", out_path)


def plot_nonparametric_regression(
    data: pd.DataFrame,
    output_folder: Path,
    y_variable: str = "positive_lost_load",
    x_variable: str = "water_level_salto",
    overwrite: bool = False
) -> None:
    out_path = output_folder / f"predicted_probability_{y_variable}.png"
    if skip_if_exists(out_path, overwrite):
        return

    if not {y_variable, x_variable}.issubset(data.columns):
        logger.warning("Data missing required columns for logistic GAM.")
        return

    data = data.sample(min(500_000, len(data)))
    mask = np.isfinite(data[x_variable]) & np.isfinite(data[y_variable])
    data = data.loc[mask]   


    X = data[x_variable].values
    y = data[y_variable].values

    gam = LogisticGAM(s(0)).fit(X, y)
    x_seq = np.linspace(X.min(), X.max(), 100)
    yhat = gam.predict_proba(x_seq)

    plt.figure()
    plt.plot(x_seq, yhat, color="blue", lw=2)
    plt.title("Predicted Probability of Lost Load (GAM)")
    plt.xlabel("Water Level (Salto)")
    plt.ylabel("Predicted Probability")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved predicted probability plot to %s", out_path)
