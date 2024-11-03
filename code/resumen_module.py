import pandas as pd
import numpy as np
import processing_module as proc
from typing import Optional
from pathlib import Path
import auxiliary
import logging

logging.basicConfig(level=logging.DEBUG)

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



BASIC_RES_OPTION = {
    'folder_key': 'sim',
    'delete_first_row': True,
}

PRODUCTION_BY_PLANT_TABLE = {
    'filename': r'resUnico*.xlt',
    'table_pattern': {
        'start': 'ENERGIAS ESPERADAS POR RECURSO EN GWh ',
        'end': 'IMPACTO'
    },
    **BASIC_RES_OPTION,
    'variables': {
    'bonete': 'HID-bonete',
    'baygorria': 'HID-baygorria',
    'palmar': 'HID-palmar',
    'salto': 'HID-salto',
    'ptigre_a': 'TER-PTigreA',
    'ctr': 'TER-CTR',
    'apr': 'TER-APR',
    'engines': 'TER-motores',
    'bio_disp': 'TER-Bio_desp',
    'bio_nodisp': 'TER-Bio_nodesp',
    'new_thermal': 'TER-new_thermal',
    'ptigre_b': 'CC-PTigreB',
    'wind': 'EOLO-eoloDeci',
    'solar': 'FOTOV-solarDeci',
    'failure_0': 'FALLA-demandaPrueba_EscFalla0',
    'failure_1': 'FALLA-demandaPrueba_EscFalla1',
    'failure_2': 'FALLA-demandaPrueba_EscFalla2',
    'failure_3': 'FALLA-demandaPrueba_EscFalla3',
    'demand': 'DEM-demandaPrueba'
}
}

PRODUCTION_BY_RESOURCE_TABLE = {
    'filename': r'resUnico*.xlt',
    'table_pattern': {
        'start': 'ENERGIAS ESPERADAS POR TIPO DE RECURSO EN GWh',
        'end': 'IMPACTO'
    },
    **BASIC_RES_OPTION,
    'variables': {
        'hydros': 'HID',
        'thermals': 'TER',
        'combined_cycle': 'CC',
        'wind': 'EOLO',
        'solar': 'FOTOV',
        'import_export': 'IMPOEXPO',
        'demand': 'DEM',
        'blackout': 'FALLA'}
}

COSTS_BY_PARTICIPANT_TABLE = {
   'key': 'costs',
   'filename': r'resUnico*.xlt',
   'table_pattern': {
       'start': 'COSTOS ESPERADOS EN MUSD ',
       'end': 'COSTO DE IMPACTOS Y DE CONTRATOS DE ENERGIA'
   },
   **BASIC_RES_OPTION,
   'variables': {
        'thermal': 'new_thermal'
   }
}
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
    data_table = proc.read_xlt(file_path, option)

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

