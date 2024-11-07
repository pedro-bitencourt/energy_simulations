"""
This module contains functions for processing data from the simulation output files.
"""
import traceback
import logging
import re
from datetime import timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# Local imports
from constants import DATETIME_FORMAT, PRODUCTION_BY_RESOURCE_TABLE, PRODUCTION_BY_PLANT_TABLE
import auxiliary

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def text_to_table(text, delete_first_row):
    """
    Converts text to a table.
    """
    def break_text_into_cells(text):
        lines = text.split('\n')
        data = [line.split('\t') for line in lines]
        data = [line for line in data if not line == ['']]
        return data

    data = break_text_into_cells(text)
    if delete_first_row:
        data = data[1:]

    data = [[s.replace('\r', '') for s in sublist] for sublist in data]
    return data


def process_dataframe(dataframe, columns_options):
    """
    Processes a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        headers = dataframe[0]
        dataframe = dataframe[1:]
        dataframe = pd.DataFrame(dataframe, columns=headers)
    # print(dataframe[:5])  # debug
    drop_cols = columns_options.get('drop_columns')
    rename_cols = columns_options.get('rename_columns')
    numeric_cols = columns_options.get('numeric_columns')
    if drop_cols:
        dataframe = dataframe.drop(columns=drop_cols)
    if rename_cols:
        dataframe = dataframe.rename(columns=rename_cols)
    if numeric_cols:
        try:
            dataframe[numeric_cols] = dataframe[numeric_cols].apply(pd.to_numeric,
                                                                    errors='coerce')
        except Exception as e:
            logging.error(f'Error {e} while converting columns to numeric.')
            traceback.print_exc(limit=10)

    dataframe = dataframe.dropna()
    return dataframe


def find_table_from_pattern(text_content, table_pattern):
    if table_pattern.get('start'):
        start_pattern = table_pattern['start']
        start_match = re.search(re.escape(start_pattern),
                                text_content, re.IGNORECASE)
        if start_match:
            table_start = start_match.start()
        else:
            print(f"{table_pattern['start']} not found ")
            return None
    else:
        table_start = 0

    if table_pattern.get('end'):
        end_match = re.search(table_pattern['end'], text_content[table_start:])
        if end_match:
            table_end = table_start + end_match.start()
        else:
            print(f"{table_pattern['end']} not found ")
            return None
    else:
        table_end = len(text_content)

    if table_end <= table_start:
        print(f"End pattern not found or invalid ")
        return None

    table_text = text_content[table_start:table_end]
    table_text = re.sub(r' +', ' ', table_text)
    return table_text


def read_xlt(file_path, options):
    """
    Reads an excel file and returns a list of lists.
    """

    def break_text_into_cells(text):
        lines = text.split('\n')
        data = [line.split('\t') for line in lines]
        data = [line for line in data if not line == ['']]
        return data

    def text_to_table(text, delete_first_row):
        data = break_text_into_cells(text)
        if delete_first_row:
            data = data[1:]
        data = [[s.replace('\r', '') for s in sublist] for sublist in data]
        return data

    # open the file as binary
    content = auxiliary.try_open_file(file_path, 'rb')

    # handle case where file could not be opened
    if not content:
        logging.error(f"Could not open {file_path}")
        return None

    # decode the content
    text_content = content.decode('latin1', errors='ignore')

    # find the table in the raw text
    table_pattern = options.get('table_pattern', {})
    table_text = find_table_from_pattern(text_content, table_pattern)

    # handle case where table could not be found
    if table_text is None:
        logging.error(f"Table not found in {file_path}")
        logging.error(f"Pattern: {table_pattern}")
        return False

    # convert the table text to a list of lists
    delete_first_row = options.get('delete_first_row', False)
    data = text_to_table(table_text, delete_first_row)
    if data[-1] == ['']:
        data = data[:-1]
    return data


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

    # extract the dataframe
    dataframe: pd.DataFrame = read_xlt(file_path=file_path, options=option)

    # process the dataframe
    columns_options = option.get('columns_options', False)
    process_function = option.get('process', None)
    if process_function:
        dataframe = option['process'](dataframe)
    if columns_options:
        dataframe = process_dataframe(dataframe, columns_options)

    return dataframe


def get_present_value(dataframe: pd.DataFrame,
                      variable: str,
                      annual_interest_rate: float,
                      log: bool = False) -> float:
    """
    Computes the present value of a variable in a pandas DataFrame.
    The DataFrame must have a 'datetime' column.
    """
    # Set reference date and calculate daily interest rate
    reference_date = pd.to_datetime('2023-01-01').normalize()
    daily_rate = (1 + annual_interest_rate)**(1/365) - 1

    # count the rows
    num_rows = dataframe.shape[0]
    # assuming a row is 1 hour, get the number of year of rows
    years = num_rows / (365 * 24)

    # log this if log is True
    if log:
        logging.debug(
            "Getting present value of variable %s for %d years", variable, years)

    # Convert datetime and calculate days difference
    days_diff = (pd.to_datetime(
        dataframe['datetime']) - reference_date).dt.days

    # Calculate discount factor and present value
    discount_factor = 1 / (1 + daily_rate)**days_diff
    present_values = pd.to_numeric(
        dataframe[variable], errors='coerce') * discount_factor

    return present_values.sum()


# Helper Functions
def process_marginal_cost(marginal_cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the marginal cost DataFrame to include a datetime column.

    Args:
        marginal_cost_df (pd.DataFrame): The marginal cost DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame with a datetime column.
    """
    def datetime_from_row(row):
        paso_start_date = pd.to_datetime(row['FechaInicialPaso'], format='%Y/%m/%d')
        hours_added = pd.to_numeric(row['Int.MuestreoDelPaso'], errors='coerce')
        return (paso_start_date + timedelta(hours=int(hours_added))).strftime(DATETIME_FORMAT)

    marginal_cost_df['datetime'] = marginal_cost_df.apply(datetime_from_row, axis=1)
    return marginal_cost_df

def open_dataframe(option: Dict[str, Any], input_folder: Path) -> Optional[pd.DataFrame]:
    """
    Opens and processes a DataFrame based on provided options.

    Args:
        option (dict): Options for reading and processing the DataFrame.
        input_folder (Path): The folder containing the input files.

    Returns:
        pd.DataFrame or None: The processed DataFrame, or None if operation fails.
    """
    filename_pattern = option.get('filename')
    if not filename_pattern:
        logging.error('Filename pattern not provided in options.')
        return None

    file_path = auxiliary.try_get_file(input_folder, filename_pattern)
    if not file_path:
        logging.error(f'File matching pattern {filename_pattern} not found in {input_folder}.')
        return None

    logging.info(f'Opening DataFrame from {file_path} with options {option}')
    dataframe = read_xlt(file_path=file_path, options=option)
    if dataframe is None:
        logging.error(f'Could not read DataFrame from {file_path}.')
        return None

    columns_options = option.get('columns_options')
    if columns_options:
        dataframe = process_dataframe(dataframe, columns_options)
        if dataframe is None:
            logging.error(f'Could not process DataFrame columns for {file_path}.')
            return None

    return dataframe


def total_production_by_plant(sim_folder: Path) -> dict:
    # load the production by plant data
    prod_by_plant = production_by_plant(sim_folder)

    if prod_by_plant is None:
        return None

    # calculate sum for each column except 'year'
    total_production = prod_by_plant.drop('year', axis=1).sum()

    # convert the result to a dictionary
    production_dict = total_production.to_dict()

    return production_dict

def total_production_by_resource(sim_folder: Path) -> dict:
    # load the production by resource data
    prod_by_resource = production_by_resource(sim_folder)

    if prod_by_resource is None:
        return None

    # calculate sum for each column except 'year'
    total_production = prod_by_resource.drop('year', axis=1).sum()

    # convert the result to a dictionary
    production_dict = total_production.to_dict()

    return production_dict


def production_by_resource(sim_folder: Path) -> Optional[pd.DataFrame]:
    prod_by_resource: Optional[pd.DataFrame] = process_res_file(PRODUCTION_BY_RESOURCE_TABLE,
                                                                sim_folder)
    return prod_by_resource

def production_by_plant(sim_folder: Path) -> Optional[pd.DataFrame]:
    prod_by_plant: Optional[pd.DataFrame] = process_res_file(PRODUCTION_BY_PLANT_TABLE,
                                                             sim_folder)
    return prod_by_plant



multipliers = {'production': 1_000, 'costs': 1_000_000}

def process_res_file(option, sim_folder):
    '''
    Processes the output from read_res_file. It converts the values to numeric and renames the columns.
    '''
    def convert_to_numeric(val):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str):
            try:
                return float(val.replace(',', '.'))
            except ValueError:
                return np.nan
        return np.nan

    raw_data = read_res_file(option, sim_folder)

    # check if the data was read correctly
    if raw_data is None:
        return None

    # convert the data to numeric
    for column in raw_data.columns:
        if column != 'year':
            # convert the column to numeric
            raw_data[column] = pd.to_numeric(raw_data[column])

    # Drop rows with all NaN values (if any resulted from the conversion)
    raw_data.dropna(how='all', inplace=True)
    # drop the last two rows
    raw_data.drop(raw_data.tail(2).index, inplace=True)

    # Create a mapping from raw column names to desired column names
    rename_map = {raw_name: desired_name for desired_name, raw_name in option['variables'].items()}
    # Rename the columns using the mapping
    raw_data.rename(columns=rename_map, inplace=True)
    # Define the columns to keep, ensuring 'year' is included
    columns_to_keep = ['year'] + list(rename_map.values())
    # Select only the desired columns
    processed_data = raw_data[columns_to_keep]

    # drop duplicated columns
    processed_data = processed_data.loc[:, ~processed_data.columns.duplicated()]
    return processed_data

def read_res_file(option, sim_folder):
    # get the file path of the resumen file
    file_path = auxiliary.try_get_file(sim_folder, option['filename'])

    # read the table from the resumen file
    data_table = read_xlt(file_path, option)

    # check if last row has length less than 2
    if len(data_table[-1]) < 2:
        data_table = data_table[:-1]


    # check if the table was read correctly:
    if not data_table:
        logging.error("Error reading the table from the file %s",
                      file_path)
        return None

    # convert the table to a DataFrame
    # create the years column
    number_of_years = len(data_table[0]) - 1
    dataframe = pd.DataFrame({'year':  [2023 + y for y in range(number_of_years)]})

    logging.debug(dataframe)


    # add the data to the DataFrame
    for line in data_table:
        # check if the line has the correct number of columns
        if len(line) != number_of_years + 1:
            logging.error("Faulty line found while reading %s",
                          file_path)
            logging.error("Faulty line:%s", line)
            continue
        dataframe[line[0]] = line[1:]

    logging.error("process_res_file output: %s", dataframe)

    return dataframe
