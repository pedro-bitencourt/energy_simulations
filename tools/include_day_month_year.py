# Filename: include_day_month_year.py
# Objective: Include day, month and year in a set of run files.

from pathlib import Path
import pandas as pd

def main():
    simulations_folder: Path = Path("/Users/pedrobitencourt/Projects/energy_simulations/simulations/")
    experiment_name: str = "qturbmax"
    experiment_raw_folder: Path = simulations_folder / experiment_name / "raw"
    for run_file in experiment_raw_folder.glob("*.csv"):
        include_day_month_year_file(run_file)

def include_day_month_year_file(run_file: Path):
    # Read the run file
    df = pd.read_csv(run_file)
    df['day'] = pd.to_datetime(df['datetime']).dt.day
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df['year'] = pd.to_datetime(df['datetime']).dt.year
    df.to_csv(run_file, index=False)


if __name__ == "__main__":
    main()

