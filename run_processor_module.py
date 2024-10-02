'''

'''
import logging
import pandas as pd
from pathlib import Path
import json
from typing import Optional
from datetime import timedelta
import auxiliary
import processing_module as proc
import participant_module as part
import resumen_module as res
from run_module import Run
from constants import DATETIME_FORMAT, MARGINAL_COST_DF, DEMAND_DF, SCENARIOS

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

WEEK_HOURS_BIN = list(range(0, 168))


class RunProcessor:
    # if run was not successfuly, do not initialize
    def __new__(cls, run: Run):
        # check if the run was successful
        if not run.successful_function():
            logging.error(f'Run {run.run_name} was not successful.')
            return None
        return super(RunProcessor, cls).__new__(cls)

    def __init__(self, run: Run):
        self.run: Run = run
        self.paths: dict = self.update_paths(run.paths)
        self.processed: bool = self.get_processed_status()

    def update_paths(self, paths: dict):
        paths['marginal_cost'] = f"{paths['output']}/marginal_cost.csv"
        paths['bash_script'] = f"{paths['input']}/{self.run.run_name}_proc.sh"
        paths['price_distribution'] = Path(
            f"{paths['output']}/price_distribution.csv")
        opt_and_sim_folders: dict = self.run.get_opt_and_sim_folders()
        sim_folder: Path = opt_and_sim_folders.get('sim', False)
        if not sim_folder.exists():
            logging.critical("sim folder was not found in %s for run %s",
                             paths['output'], self.run.run_name)
        paths.update(opt_and_sim_folders)
        return paths

    def results_run(self):
        # check if the run was successful
        if not self.run.successful:
            logging.error(f'Run {self.run.run_name} was not successful.')
            return None

        # get the variable values
        variable_values: dict = {var: var_dict['value']
                                 for var, var_dict in self.run.variables.items()}

        # create a header for the results
        header: dict = {'run_name': self.run.run_name,
                        **variable_values}

        # get the price results
        price_results: dict = self.get_price_results()

        # get the production results
        production_results: dict = self.get_production_results()

        # concatenate the results
        results: dict = {**header, **price_results, **production_results}

        # save the results to a json file
        with open(self.paths['results_json'], 'w') as file:
            json.dump(results, file, indent=4)

        return results

    def get_processed_status(self):
        if self.paths['results_json'].exists():
            return True
        return False

    #@auxiliary.cache(lambda self: self.paths['marginal_cost'])
    def extract_marginal_costs_df(self) -> Optional[pd.DataFrame]:
        '''
        Extracts the marginal cost dataframe from the simulation folder.
        Resulting dataframe must have:
        - a 'datetime' column in the format DATETIME_FORMAT
        - a 'SCENARIOS' column with the marginal costs for each scenario
        '''
        # extract the marginal cost dataframe
        marginal_cost_df = open_dataframe(MARGINAL_COST_DF,
                                          self.paths['sim'])

        # check if the marginal cost dataframe was extracted successfully
        if marginal_cost_df is None:
            logging.error('Marginal cost dataframe could not be extracted.')
            return None

        # check if the marginal cost dataframe has NaN values
        if marginal_cost_df.isna().sum().sum() > 0:
            logging.error('Marginal cost dataframe contains NaN values.')
            return None

        # log the head of the marginal cost dataframe for debugging purposes
        logging.debug('marginal cost head: %s', marginal_cost_df.head())

        # process the marginal cost dataframe
        marginal_cost_df = process_marginal_cost(
            marginal_cost_df)

        # check if any NaN values are present in the dataframe
        if marginal_cost_df.isna().sum().sum() > 0:
            logging.error('Marginal cost dataframe contains NaN values.')

            # find the instances of NaN values and log their data
            nan_rows = marginal_cost_df[marginal_cost_df.isna().any(axis=1)]
            logging.error(f'{nan_rows=}')
            logging.error(f'{marginal_cost_df.head()}')
            return None

        # check if the 'datetime' column exists
        if 'datetime' not in marginal_cost_df.columns:
            logging.error(
                'Marginal cost dataframe does not contain a datetime column.')
            return None

        # force datetime column to be in DATETIME_FORMAT
        marginal_cost_df['datetime'] = pd.to_datetime(
            marginal_cost_df['datetime'], format=DATETIME_FORMAT)
        
        # save marginal cost dataframe
        marginal_cost_df.to_csv(self.paths['marginal_cost'], index=False)
        return marginal_cost_df

#    @auxiliary.cache(lambda self: self.paths['price_distribution'])
    def get_price_distribution(self) -> Optional[pd.DataFrame]:
        # load the price data
        price_df: Optional[pd.DataFrame] = self.extract_marginal_costs_df()

        if price_df is None:
            logging.error('Price dataframe could not be extracted.')
            return None

        # get the average price for each hour, aggregating over scenarios
        avg_price_df = price_df[['datetime']].copy()
        # check if the run was successful
        avg_price_df['price_avg'] = price_df[SCENARIOS].mean(axis=1)

        # check if the 'datetime' column exists and is a datetime object
        if 'datetime' not in avg_price_df.columns:
            logging.error(
                'Price dataframe does not contain a datetime column.')
            return None

        if not pd.api.types.is_datetime64_any_dtype(avg_price_df['datetime']):
            logging.error('Price dataframe does not have a datetime column.')
            return None

        # create a column with the hour of the week
        avg_price_df['hour_of_week'] = avg_price_df['datetime'].dt.dayofweek * \
            24 + avg_price_df['datetime'].dt.hour

        # create a column with the bin of the hour of the week
        avg_price_df['hour_of_week_bin'] = pd.cut(
            avg_price_df['hour_of_week'], bins=WEEK_HOURS_BIN)

        # compute the average price for each bin
        price_distribution = avg_price_df.groupby(
            'hour_of_week_bin', as_index=False)['price_avg'].mean()

        price_distribution.columns = ['hour_of_week_bin', 'price_avg']

        return price_distribution

    def get_production_results(self) -> dict:
        # initialize results dictionary
        production_results: dict = {}

        # get total production by resource
        production_by_resource: dict[str, float] = res.total_production_by_resource(
            self.paths['sim'])

        # append the production by resource to the results dictionary
        production_results.update(
            {f'total_production_{resource}': production_by_resource[resource]
                for resource in production_by_resource})

        # get total production by plant
        production_by_plant: dict[str, float] = res.total_production_by_plant(
            self.paths['sim'])

        # get the total production for the new thermal plant
        new_thermal_production: float = production_by_plant.get(
            'new_thermal', 0)

        # append new_thermal_production to the results dictionary
        production_results['new_thermal_production'] = new_thermal_production

        return production_results

    def get_price_results(self):
        # load the price data
        price_df = self.extract_marginal_costs_df()

        # load the demand data
        demand_df = open_dataframe(DEMAND_DF, self.paths['sim'])

        if price_df is None or demand_df is None:
            logging.error('Price or demand dataframe could not be extracted.')
            return None

        # get the simple average of the price over both axis
        price_avg: float = price_df[SCENARIOS].mean(axis=1).mean(axis=0)

        # get the weighted average of the price
        price_times_demand_df: pd.DataFrame = price_df[SCENARIOS] * \
            demand_df[SCENARIOS]
        price_weighted_avg: float = price_times_demand_df.sum().sum() / \
            demand_df[SCENARIOS].sum().sum()

        # avoid memory leaks
        del price_df, demand_df, price_times_demand_df

        results: dict = {'price_avg': price_avg,
                         'price_weighted_avg': price_weighted_avg}

        return results

    def get_profits(self):
        if not self.run.successful_function():
            logging.warning("Run %s was not succesfully computed",
                            self.run.run_name)
            return None

        logging.info('Extracting profits for %s', self.paths['sim'])

        # extract the marginal costs
        marginal_cost_df: pd.DataFrame = self.extract_marginal_costs_df()

        # free memory
        del marginal_cost_df

        # hardcoded for now
        renewables = ['wind', 'solar']
        capacities = {'wind': self.run.variables['wind']['value'],
                      'solar': self.run.variables['solar']['value']}

        # iterate over the renewables
        profits = {}
        for renewable in renewables:
            # load the participant
            participant_capacity = capacities[renewable]
            participant_object = part.Participant(
                renewable, participant_capacity, self.paths)

            logging.info('Processing participant %s with capacity %s.',
                         renewable, participant_capacity)

            # process the participant
            results_participant = participant_object.process_participant(
                self.run.general_parameters['daily'])

            # check whether the participant was processed successfully
            if results_participant is not None:
                logging.info(
                    'Participant %s processed successfully.', renewable)
                logging.debug(f'{results_participant=}')

                # update the profits dictionary
                profits[renewable] = results_participant[f'profits_{renewable}']
            else:
                logging.critical(
                    f'Could not process {renewable} for run {self.run.run_name}')

        return profits

    def create_bash_script(self):
        # create the data dictionary
        run_data = {
            'input_folder': str(self.paths['input']),
            'general_parameters': self.run.general_parameters,
            'variables': self.run.variables,
            'output_folder': str(self.paths['output'])
        }
        run_data = json.dumps(run_data)

        requested_time: str = '0:30:00'

        # write the bash script
        with open(self.paths['bash_script'], 'w') as f:
            f.write(f'''#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time={requested_time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --job-name=proc_{self.run.run_name}
#SBATCH --output={self.paths['output']}/{self.run.run_name}_proc.out
#SBATCH --error={self.paths['output']}/{self.run.run_name}_proc.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0

python - <<END

import sys
import json
sys.path.append('/projects/p32342/code')
from run_module import Run 
from run_processor_module import RunProcessor

print('Processing run {self.run.run_name}...')
sys.stdout.flush()
sys.stderr.flush()

# load the run data
run_data = {json.loads(run_data)}
# create Run object
run = Run(**run_data)
# create RunProcessor object
run_processor = RunProcessor(run)
# extract results
results = run_processor.results_run()

# extract price distribution
price_distribution = run_processor.get_price_distribution()
if price_distribution is not None:
    price_distribution.to_csv(run_processor.paths['price_distribution'], index=False)

sys.stdout.flush()
sys.stderr.flush()
END
''')

        return self.paths['bash_script']


def process_marginal_cost(marginal_cost_df: pd.DataFrame) -> pd.DataFrame:
    # log the head of marginal_cost_df for debugging purposes
    logging.debug(f'{marginal_cost_df.head()=}')

    def datetime_from_row_marginal_cost(row):
        """
        Constructs a datetime column from a row.
        """
        paso_start_date = pd.to_datetime(
            row['FechaInicialPaso'], format='%Y/%m/%d')
        hours_added = pd.to_numeric(
            row['Int.MuestreoDelPaso'], errors='coerce')
        return (paso_start_date + timedelta(hours=int(hours_added))).strftime(DATETIME_FORMAT)

    # convert the 'FechaInicialPaso' and 'Int.MuestreoDelPaso' columns to datetime
    marginal_cost_df['datetime'] = marginal_cost_df.apply(datetime_from_row_marginal_cost,
                                                          axis=1)
    return marginal_cost_df


def get_present_value(dataframe: pd.DataFrame,
                      variable: str,
                      annual_interest_rate: float) -> float:
    """
    Computes the present value of a variables in a pandas DataFrame.
    Note: the DataFrame must have a 'datetime' column.
    """
    # calculate the daily interest rate
    daily_interest_rate = (1 + annual_interest_rate)**(1 / 365) - 1
    # Set the reference date and adjust it to 00:00 of that day
    # Adjusts to 00:00 of 2023-01-01
    reference_date = pd.to_datetime('2023-01-01').normalize()

    # ensure the 'datetime' column is parsed correctly to datetime objects
    dataframe['datetime'] = pd.to_datetime(
        dataframe['datetime'], errors='coerce')

    # create a column with delta_days from the reference date
    dataframe['delta_days'] = (dataframe['datetime'] - reference_date).dt.days

    # Create a column with the discount factor
    dataframe['discount_factor'] = 1 / \
        (1 + daily_interest_rate)**(dataframe['delta_days'])

    dataframe[variable] = pd.to_numeric(dataframe[variable], errors='coerce')
    nan_count = dataframe[variable].isna().sum()
    if nan_count > 100:
        print(
            f'CRITICAL ERROR: NaN count = {nan_count} when getting present value of {variable}')
        print(f'{dataframe.head()=}')

    # Calculate the product of the variables and the discount factor
    dataframe['product'] = dataframe[variable] * dataframe['discount_factor']
    result = dataframe['product'].sum()
    # Return the sum of the product column
    return result


def open_dataframe(option: dict, input_folder: str) -> Optional[pd.DataFrame]:
    '''
    Extracts a dataframe 
    '''
    # check if input file exists
    file_path = auxiliary.try_get_file(input_folder,
                                       option['filename'])

    # handle case where input file is not found
    if not file_path:
        logging.error('File %s could not be found in folder %s.',
                      file_path, input_folder)
        return None

    filename = option['filename']
    if not file_path:
        logging.error(f'{option} does not contain a filename key.')
        return None

    # log the operation
    logging.debug(f'Extracting dataframe from {file_path}')
    logging.debug(f'{option=}')

    # extract the dataframe
    dataframe = proc.read_xlt(
        file_path=file_path, options=option)

    # process the dataframe
    columns_options = option.get('columns_options', False)
    if columns_options:
        dataframe = proc.process_dataframe(dataframe, columns_options)

    # check if the dataframe was processed successfully
    if dataframe is None:
        logging.error('%s could not be processed.', filename)

    return dataframe
