"""
File name: run_processor_module.py

Description:
This module implements the RunProcessor class, which extends the Run class to process run results.
It extracts data such as marginal costs, price distributions, production results, and profits.

Public methods:
- RunProcessor: Initializes the RunProcessor with an existing Run instance.
- process_run: Processes the run locally or submits a processing job to the cluster.
- get_profits: Computes profits for the specified endogenous variables.
- submit_processor_job: Submits a job to process the run on a cluster.
"""

import logging
from pathlib import Path
import json
from typing import Optional, Dict
import numpy as np
import pandas as pd

import src.auxiliary
import src.processing_module as proc
from src.participant_module import Participant
from src.run_module import Run
from src.constants import MARGINAL_COST_DF, DEMAND_DF, SCENARIOS

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WEEK_HOURS_BIN = list(range(0, 169))  # Adjusted to include 168


class RunProcessor(Run):
    """
    Extends the Run class to extract data from MOP's output, such as:
        - The simulated marginal cost for each scenario, over the timeline
        - The average intraweek price distribution 
        - Total production by resource
        - Profits for each endogenous capacity

    Attributes:
        All attributes inherited from Run class
        paths (dict): Extended dictionary of relevant paths for processing
    """

    def __init__(self, run: Run):
        """
        Initializes RunProcessor with an existing Run instance.

        Args:
            run (Run): An existing Run instance to process

        Raises:
            ValueError: If the run was not successful
        """
        super().__init__(
            parent_folder=run.paths['parent_folder'],
            general_parameters=run.general_parameters,
            variables=run.variables
        )

        if not self.successful(log=True):
            logger.error(f'Run {self.name} was not successful.')
            raise ValueError(f'Run {self.name} was not successful.')

        self._update_paths()
        # Create dict of participants
        list_participants = ['wind', 'solar', 'thermal']
        self.participants_dict: dict[str, Participant] = {var: Participant(var,
                                                                           self.variables[var]['value'],
                                                                           self.paths,
                                                                           run.general_parameters)
                                                          for var in list_participants}

    def production_participant(self, participant_key: str) -> pd.DataFrame:
        production_df: pd.DataFrame = self.participants_dict[participant_key].production_df(
        )
        return production_df

    def variable_costs_participant(self, participant_key: str) -> pd.DataFrame:
        costs_df: pd.DataFrame = self.participants_dict[participant_key].variable_costs_df(
        )
        return costs_df

    def water_level_participant(self, participant_key: str) -> pd.DataFrame:
        water_level_df: pd.DataFrame = self.participants_dict[participant_key].water_level_df(
        )
        return water_level_df

    def _update_paths(self) -> None:
        """
        Updates the paths dictionary with additional paths needed for processing.
        """
        self.paths['marginal_cost'] = self.paths['folder'] / \
            'marginal_cost.csv'
        self.paths['bash_script'] = self.paths['folder'] / \
            f'{self.name}_proc.sh'
        self.paths['price_distribution'] = self.paths['folder'] / \
            'price_distribution.csv'
        self.paths['results_json'] = self.paths['folder'] / 'results.json'

    def process(self, process_locally: bool = True) -> None:
        """
        Processes the run and extracts results, saving them to a JSON file.
        If process_locally is False, submits a job to process the run on the cluster.

        Args:
            process_locally (bool): If True, processes the run locally. If False, submits a job to the cluster.

        Returns:
            dict or None: The results dictionary if processed locally and successful, None otherwise.
        """
        if process_locally:
            logger.info(f'Processing run {self.name} locally.')
            try:
                # Get variable values
                variable_values = {var: var_dict['value']
                                   for var, var_dict in self.variables.items()}

                # Create a header for the results
                header = {'run_name': self.name, **variable_values}

                logger.debug("Getting price results for run %s", self.name)
                # Get price results
                price_results = self._get_price_results()

                logger.debug("Saving price distribution for run %s", self.name)
                # Get price distribution
                price_distribution = self._intraweek_price_distribution()
                # Save price distribution
                price_distribution.to_csv(
                    self.paths['price_distribution'], index=False)
                logger.debug(
                    "Getting production results for run %s", self.name)
                # Get production results
                production_results = self._get_production_results()

                # Concatenate results
                results = {**header, **price_results, **production_results}

                # Convert numpy types to native Python types for JSON serialization
                results = src.auxiliary.convert_numpy_types(results)

                # Save results to JSON
                with open(self.paths['results_json'], 'w') as file:
                    json.dump(results, file, indent=4)

                logger.info(
                    f'Results for run {self.name} saved successfully.')
            except Exception as e:
                logger.error(
                    f'Error processing results for run {self.name}: {e}')
        else:
            logger.info(
                f'Submitting processing job for run {self.name} to the cluster.')
            self.submit_processor_job()

    def results(self):
        """
        Loads the results from the JSON file.
        """
        with open(self.paths['results_json'], 'r') as file:
            results = json.load(file)
        return results

    def marginal_cost_df(self) -> pd.DataFrame:
        """
        Extracts the marginal cost DataFrame from the simulation folder.

        Returns:
            pd.DataFrame or None: The marginal cost DataFrame, or None if extraction fails.
        """
        # Extract marginal cost DataFrame
        marginal_cost_df = proc.open_dataframe(
            MARGINAL_COST_DF, self.paths['sim'])

        # Save marginal cost DataFrame
        marginal_cost_df.to_csv(self.paths['marginal_cost'], index=False)
        logger.info(
            f'Marginal cost DataFrame saved to {self.paths["marginal_cost"]}')
        return marginal_cost_df

    def _intraweek_price_distribution(self) -> pd.DataFrame:
        """
        Computes and returns the price distribution DataFrame.

        Returns:
            pd.DataFrame or None: The price distribution DataFrame, or None if computation fails.
        """
        price_df = self.marginal_cost_df()
        # Compute average price across scenarios
        price_df['price_avg'] = price_df[SCENARIOS].mean(axis=1)

        # Add hour of the week
        price_df['hour_of_week'] = price_df['datetime'].dt.dayofweek * \
            24 + price_df['datetime'].dt.hour

        # Bin hours into weekly bins
        price_df['hour_of_week_bin'] = pd.cut(price_df['hour_of_week'],
                                              bins=WEEK_HOURS_BIN, right=False)

        # Compute average price per bin
        price_distribution = price_df.groupby('hour_of_week_bin', as_index=False)[
            'price_avg'].mean()

        # Save price distribution
        price_distribution.to_csv(
            self.paths['price_distribution'], index=False)
        logger.info(
            f'Price distribution saved to {self.paths["price_distribution"]}')
        return price_distribution

    def _get_production_results(self) -> Dict[str, float]:
        """
        Extracts production results from the simulation data.

        Returns:
            dict: A dictionary containing total production by resource and new thermal production.
        """
        production_results: dict = {}

        # Get total production by resource
        production_by_resource = proc.total_production_by_resource(
            self.paths['sim'])
        production_results.update({
            f'total_production_{resource}': production_by_resource.get(resource, 0.0)
            for resource in production_by_resource
        })

        return production_results

    def _get_price_results(self) -> Dict[str, float]:
        """
        Computes price results from marginal cost and demand data.

        Returns:
            dict or None: A dictionary containing price averages, or None if computation fails.
        """
        price_df = self.marginal_cost_df()
        demand_df = proc.open_dataframe(DEMAND_DF, self.paths['sim'])

        # Compute simple average price
        price_avg = price_df[SCENARIOS].mean().mean()

        # Compute weighted average price
        price_times_demand = price_df[SCENARIOS] * demand_df[SCENARIOS]
        price_weighted_avg = price_times_demand.values.sum() / \
            demand_df[SCENARIOS].values.sum()

        return {'price_avg': price_avg, 'price_weighted_avg': price_weighted_avg}

    def submit_processor_job(self) -> Optional[str]:
        """
        Submits a job to process the run on a cluster.

        Returns:
            int or None: The job ID if submission is successful, None otherwise.
        """
        bash_script = self._create_bash_script()
        job_id = src.auxiliary.submit_slurm_job(bash_script)
        if job_id:
            logger.info(
                f'Processing job for run {self.name} submitted with job ID {job_id}')
        else:
            logger.error(
                f'Failed to submit processing job for run {self.name}')
        return job_id

    def get_profits(self) -> Dict[str, float]:
        """
        Computes profits for the specified endogenous variables.

        Args:
            endogenous_variables_names (list): List of endogenous variable names.

        Returns:
            dict: A dictionary of profits.
        """
        # Extract marginal cost
        marginal_cost_df: pd.DataFrame = self.marginal_cost_df()

        profits = {}
        for participant in self.participants_dict.values():
            logger.debug(
                f"Computing profits for {participant.key} participant.")
            # Compute profit for the participant
            profit: float = participant.profit(marginal_cost_df)

            logger.debug('Profit for participant %s: %s',
                         participant.key, profit)

            # Add profit to the dictionary
            profits[participant.key] = profit
        return profits

    def processed_status(self):
        if self.paths['results_json'].exists():
            return True
        return False

    def _create_bash_script(self) -> Path:
        """
        Creates a bash script to process the run on a cluster.

        Returns:
            Path: The path to the created bash script.
        """
        script_path = self.paths['bash_script']
        run_data = {
            'folder': str(self.paths['folder']),
            'general_parameters': self.general_parameters,
            'variables': self.variables
        }
        run_data_json = json.dumps(run_data)

        requested_time = '0:30:00'

        script_content = f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --job-name=proc_{self.name}
#SBATCH --output={self.paths['folder']}/{self.name}_proc.out
#SBATCH --error={self.paths['folder']}/{self.name}_proc.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0

python - <<END

import sys
import json
from pathlib import Path
sys.path.append('/projects/p32342/code')
from run_processor_module import RunProcessor

print('Processing run {self.name}...')
sys.stdout.flush()
sys.stderr.flush()

# Load the run data
run_data = {run_data_json}
# Create RunProcessor object
run_processor = RunProcessor(
    run=Run(
        folder=Path(run_data['folder']),
        general_parameters=run_data['general_parameters'],
        variables=run_data['variables']
    )
)
# Process run
results = run_processor.process_run(process_locally=True)

# Extract price distribution
price_distribution = run_processor._intraweek_price_distribution()
price_distribution.to_csv(run_processor.paths['price_distribution'], index=False)

sys.stdout.flush()
sys.stderr.flush()
END
'''

        # Write the bash script
        with open(script_path, 'w') as f:
            f.write(script_content)

        logger.info(f'Bash script created at {script_path}')
        return script_path
