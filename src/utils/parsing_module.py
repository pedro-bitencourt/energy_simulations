"""
This module contains functions for processing data from the simulation output files.
"""
import traceback
import logging
import re
from datetime import timedelta
from typing import Dict, Any
from pathlib import Path
import pandas as pd

# Local imports
from src.constants import DATETIME_FORMAT, POSTE_FILEPATH
from src.utils import auxiliary


logger = logging.getLogger(__name__)


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
            logger.error(f'Error {e} while converting columns to numeric.')
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
        logger.error(f"Could not open {file_path}")
        return None

    # decode the content
    text_content = content.decode('latin1', errors='ignore')

    # find the table in the raw text
    table_pattern = options.get('table_pattern', {})
    table_text = find_table_from_pattern(text_content, table_pattern)

    # handle case where table could not be found
    if table_text is None:
        logger.error(f"Table not found in {file_path}")
        logger.error(f"Pattern: {table_pattern}")
        return False

    # convert the table text to a list of lists
    delete_first_row = options.get('delete_first_row', False)
    data = text_to_table(table_text, delete_first_row)
    if data[-1] == ['']:
        data = data[:-1]
    return data


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
        paso_start_date = pd.to_datetime(
            row['FechaInicialPaso'], format='%Y/%m/%d')
        hours_added = pd.to_numeric(
            row['Int.MuestreoDelPaso'], errors='coerce')
        return (paso_start_date + timedelta(hours=int(hours_added))).strftime(DATETIME_FORMAT)

    marginal_cost_df['datetime'] = marginal_cost_df.apply(
        datetime_from_row, axis=1)
    return marginal_cost_df


def open_dataframe(option: Dict[str, Any], input_folder: Path,
                   daily: bool = True) -> pd.DataFrame:
    """
    Opens and processes a DataFrame based on provided options.

    Args:
        option (dict): Options for reading and processing the DataFrame.
        input_folder (Path): The folder containing the input files.

    Returns:
        pd.DataFrame or None: The processed DataFrame, or None if operation fails.
    """
    # Validate the option
    list_of_keys = ['name', 'filename', 'columns_options']
    if not all(key in option for key in list_of_keys):
        logger.error('Option does not contain all required keys.')
        raise ValueError(
            f'Option {option} does not contain all required keys.')

    # Get the file path; raise error if not found
    file_path = auxiliary.try_get_file(input_folder, option['filename'])
    if not file_path:
        logger.error(
            f'File matching pattern {option["filename"]} not found in {input_folder}.')
        raise FileNotFoundError(
            f'File matching pattern {option["filename"]} not found in {input_folder}.')

    # Read the DataFrame; raise error if not found
    data = read_xlt(file_path=file_path, options=option)
    if data is None:
        logger.error(f'Could not read DataFrame from {file_path}.')
        logger.debug(f'{data=}')
        raise ValueError(f'Could not read DataFrame from {file_path}.')

    # Process the DataFrame columns
    dataframe = process_dataframe(data, option['columns_options'])

    if dataframe is None:
        logger.error('Error processing DataFrame %s', dataframe)
        raise ValueError('Error processing DataFrame %s', dataframe)
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

    # Create the datetime column
    if option.get('convert_poste', True):
        if option['name'] == 'marginal_cost':
            dataframe = process_marginal_cost(dataframe)
        else:
            dataframe = convert_from_poste_to_datetime(dataframe, daily)

    columns_to_keep = ['datetime'] + [str(i) for i in range(114)]
    dataframe = dataframe[columns_to_keep]

    # Validate the dataframe
    validate_dataframe(dataframe, option['name'])
    #logger.debug(f'{dataframe.head()=}')

    return dataframe

def validate_dataframe(dataframe: pd.DataFrame, name: str) -> bool:
    # Ensure datetime is in the correct format
    if 'datetime' in dataframe.columns:
        try:
            dataframe['datetime'] = pd.to_datetime(
                dataframe['datetime'], format=DATETIME_FORMAT)
        except ValueError:
            dataframe['datetime'] = pd.to_datetime(
                dataframe['datetime'], format="%Y/%m/%d/%H:%M:%S")
            dataframe['datetime'] = dataframe['datetime'].dt.strftime(
                DATETIME_FORMAT)
            dataframe['datetime'] = pd.to_datetime(
                dataframe['datetime'], format=DATETIME_FORMAT)
    else:
        logger.error(
            'DataFrame %s does not contain a datetime column.', name)
        raise ValueError(
            f'DataFrame {name} does not contain a datetime column.')

    # Check for NaN values
    if dataframe.isna().sum().sum() > 0:
        logger.error('DataFrame %s contains NaN values.', name)
        nan_rows = dataframe[dataframe.isna().any(
            axis=1)]
        logger.error(f'Rows with NaN values:\n{nan_rows}')
        raise ValueError(
            'Marginal cost DataFrame contains NaN values.')

    return True

def convert_from_poste_to_datetime(participant_df: pd.DataFrame,
                                   daily: bool = True) -> pd.DataFrame:
    """
    Converts 'poste' time format to datetime.
    """
    if daily:
        if participant_df.columns[-1] == "paso_start" and "paso_start" in participant_df.columns[:-1]:
            participant_df = participant_df.iloc[:, :-1]

        participant_df["paso_start"] = pd.to_datetime(
            participant_df["paso_start"], format="%Y/%m/%d/%H:%M:%S"
        )
        participant_df["datetime"] = participant_df.apply(
            lambda row: row["paso_start"] + timedelta(hours=float(row["poste"])), axis=1
        )
        return participant_df
    return convert_from_poste_to_datetime_weekly(participant_df)


def convert_from_poste_to_datetime_weekly(participant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts 'poste' time format to datetime for weekly data.
    """
    poste_dict_df = pd.read_csv(POSTE_FILEPATH)
    scenario_columns = [col for col in poste_dict_df.columns if col.isdigit()]
    poste_dict_long = pd.melt(
        poste_dict_df,
        id_vars=["paso", "paso_start", "datetime"],
        value_vars=scenario_columns,
        var_name="scenario",
        value_name="poste",
    )

    poste_dict_long = poste_dict_long.astype({"paso": int, "poste": int})
    participant_df = participant_df.astype({"paso": int, "poste": int})
    poste_dict_long["datetime"] = pd.to_datetime(
        poste_dict_long["datetime"], format=DATETIME_FORMAT
    )

    participant_long = pd.melt(
        participant_df, id_vars=["paso", "poste"], var_name="scenario", value_name="value"
    )

    result = pd.merge(
        participant_long,
        poste_dict_long,
        on=["paso", "poste", "scenario"],
        how="left",
    )

    result = result.dropna(subset=["datetime", "value"])
    result = result.sort_values(["datetime", "scenario"])
    result = result.drop_duplicates(
        subset=["datetime", "scenario"], keep="first")

    final_result = result.pivot(
        index="datetime", columns="scenario", values="value")
    final_result = final_result.sort_index()
    final_result["datetime"] = final_result.index

    return final_result
