import traceback
import logging
import re
import sys
import auxiliary
import pandas as pd
from datetime import datetime, timedelta
# Global variables
READING_ERROR = -12


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
