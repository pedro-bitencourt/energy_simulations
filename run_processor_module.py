'''

'''
import logging
import pandas as pd
from pathlib import Path
from datetime import timedelta
import auxiliary
import processing_module as proc
import participant_module as part
import resumen_module as res
from run_module import Run
from constants import DATETIME_FORMAT, PATTERNS_LIST, MARGINAL_COST_DF_CONFIG

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RunProcessor:
    def __init__(self, run: Run):
        self.run: Run = run
        self.paths: dict = self.update_paths(run.paths)

    def update_paths(self, paths: dict):
        paths['marginal_cost'] = f"{paths['output']}/marginal_cost.csv"
        paths.update(self.run.get_opt_and_sim_folders())
        return paths

    def extract_from_patterns(self):
        # iterate over the patterns list
        for row in PATTERNS_LIST:
            # construct the file path
            if row['complement']:
                file_path = auxiliary.try_get_file(self.paths[row['paths_key']],
                                                   row['complement'],
                                                   error_message=False)
            else:
                file_path = self.paths[row['paths_key']]
            # check if the pattern is in the file; if the patter is a regex,
            # get the first match
            result = auxiliary.find_pattern_in_file(file_path,
                                                    row['pattern'],
                                                    error_message=False)
        return result

    def process_resumen_files(self):
        for res_file in res.RES_FILES_RESULTS:
            current_value = res_file.get('current_value', False)
            if current_value:
                results = res.current_values_from_res_table(
                    res_file, self.paths['sim'])
            else:
                results = res.present_values_from_res_table(
                    res_file, self.paths['sim'])
            print(f'{results=}')
        return results

    def extract_marginal_costs_df(self):
        # extract the marginal cost dataframe
        marginal_cost_df = extract_dataframe(MARGINAL_COST_DF_CONFIG,
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
        logging.debug(f'{marginal_cost_df.head()}')

        # process the marginal cost dataframe
        marginal_cost_df: pd.DataFrame = process_marginal_cost(
            marginal_cost_df)

        # check if any NaN values are present in the dataframe
        if marginal_cost_df.isna().sum().sum() > 0:
            logging.error('Marginal cost dataframe contains NaN values.')

            # find the instances of NaN values and log their data
            nan_rows = marginal_cost_df[marginal_cost_df.isna().any(axis=1)]
            logging.error(f'{nan_rows=}')
            logging.error(f'{marginal_cost_df.head()}')
            return None

        return marginal_cost_df

    def get_profits(self):
        logging.info('Extracting profits for %s', self.paths['sim'])

        # extract the marginal costs
        marginal_cost_df: pd.DataFrame = self.extract_marginal_costs_df()

        # save the marginal costs to a csv file
        marginal_cost_df.to_csv(self.paths['marginal_cost'], index=False)

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
                logging.info(f'{results_participant=}')
                profits[renewable] = results_participant[f'profits_{renewable}']
                logging.info(f'{profits[renewable]=}')
            else:
                print(f'Error processing participant {renewable}')

        return profits


def process_marginal_cost(marginal_cost_df: pd.DataFrame) -> pd.DataFrame:
    #    # log the head of marginal_cost_df for debugging purposes
    #    logging.debug(f'{marginal_cost_df.head()=}')
    #
    #    # drop the first row of the dataframe and set it as the new columns
    #    new_columns = marginal_cost_df.iloc[0]
    #    logging.debug(f'{new_columns=}')
    #    marginal_cost_df = marginal_cost_df.iloc[1:].reset_index(drop=True)
    #    marginal_cost_df.columns = new_columns

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
        sys.exit(1)

    # Calculate the product of the variables and the discount factor
    dataframe['product'] = dataframe[variable] * dataframe['discount_factor']
    jkk
    result = dataframe['product'].sum()
    # Return the sum of the product column
    return result


def extract_dataframe(option: dict, input_folder: str):
    # check if input file exists
    file_path = auxiliary.try_get_file(input_folder,
                                       option['filename'])

    # handle case where input file is not found
    if not file_path:
        logging.error(
            f'{option["filename"]} could not be found in folder {input_folder}.')
        return None

    filename = option['filename']
    if not file_path:
        logging.error(f'{option} does not contain a filename key.')
        return None

    # log the operation
    logging.debug(f'Extracting dataframe from {file_path}')
    logging.debug(f'{option=}')

    # extract the dataframe
    dataframe: pd.DataFrame = proc.read_xlt(
        file_path=file_path, options=option)

    # process the dataframe
    columns_options = option.get('columns_options', False)

    if columns_options:
        dataframe = proc.process_dataframe(dataframe, columns_options)

    return dataframe
