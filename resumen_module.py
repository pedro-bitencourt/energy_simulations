import pandas as pd
import numpy as np
import processing_module as proc
import auxiliary

INTEREST_RATES = {'discounted': 0.1, 'undiscounted': 0}

BASIC_RES_OPTION = {
    'folder_key': 'sim',
    'delete_first_row': True,
}

PRODUCTION_BY_RESOURCE_TABLE = {
    'key': 'production_by_resource',
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
REVENUES_BY_RESOURCE_TABLE = {
    'key': 'revenues_discounted',
    'filename': r'resUnico*.xlt',
    'table_pattern': {
        'start':  'VALOR MEDIO AL COSTO MARGINAL SIN TOPE(USD/MWh)',
        'end': 'RESUMEN DE HORAS DE FALLA',
    },
    **BASIC_RES_OPTION,
    'variables': {
        'hydros': ['bonete', 'baygorria', 'palmar', 'salto'],
        'thermals': ['PTigreA', 'CTR', 'APR', 'motores',
                     'Bio_desp', 'Bio_nodesp', 'thermal_new', 'PTigreB'],
        'wind': 'eoloDeci',
        'solar': 'solarDeci',
        'demand': 'demandaPrueba',
        'import_export': ['impoEstacArg', 'impoBraRiv', 'impoBrMelo',
                          'excedentes', 'sumidero', 'expLago'],
        'blackout': ['demandaPrueba_EscFalla0', 'demandaPrueba_EscFalla1',
                     'demandaPrueba_EscFalla2', 'demandaPrueba_EscFalla3'],
        'bonete': 'bonete',
        'baygorria': 'baygorria',
        'palmar': 'palmar',
        'salto': 'salto',
        'PTigreA': 'PTigreA',
        'CTR': 'CTR',
        'APR': 'APR',
        'motores': 'motores',
        'Bio_desp': 'Bio_desp',
        'Bio_nodesp': 'Bio_nodesp',
        'thermal_new': 'thermal_new',
        'PTigreB': 'PTigreB',
        'impoEstacArg': 'impoEstacArg',
        'impoBraRiv': 'impoBraRiv',
        'impoBrMelo': 'impoBrMelo',
        'excedentes': 'excedentes',
        'sumidero': 'sumidero',
        'expLago': 'expLago',
        'demandaPrueba_EscFalla0': 'demandaPrueba_EscFalla0',
        'demandaPrueba_EscFalla1': 'demandaPrueba_EscFalla1',
        'demandaPrueba_EscFalla2': 'demandaPrueba_EscFalla2',
        'demandaPrueba_EscFalla3': 'demandaPrueba_EscFalla3',
    }
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
        'thermal': ['PTigreA', 'CTR', 'APR', 'motores', 'Bio_desp',
                    'Bio_nodesp', 'thermal_new', 'PTigreB'],
        'PTigreA': 'PTigreA',
        'CTR': 'CTR',
        'APR': 'APR',
        'motores': 'motores',
        'Bio_desp': 'Bio_desp',
        'Bio_nodesp': 'Bio_nodesp',
        'thermal_new': 'thermal_new',
        'PTigreB': 'PTigreB',
        'impoEstacArg': 'impoEstacArg',
        'impoBraRiv': 'impoBraRiv',
        'impoBrMelo': 'impoBrMelo',
        'excedentes': 'excedentes',
        'sumidero': 'sumidero',
        'expLago': 'expLago',
        'demandaPrueba_EscFalla0': 'demandaPrueba_EscFalla0',
        'demandaPrueba_EscFalla1': 'demandaPrueba_EscFalla1',
        'demandaPrueba_EscFalla2': 'demandaPrueba_EscFalla2',
        'demandaPrueba_EscFalla3': 'demandaPrueba_EscFalla3'
    }
}

CAPACITIES = {
    'key': 'capacities',
    'filename': r'resUnico*.xlt',
    'table_pattern': {
        'start': 'POTENCIA TOTAL DE GENERADORES(MW)',
        'end': 'ENERGIAS ESPERADAS POR TIPO DE RECURSO EN GWh'
    },
    **BASIC_RES_OPTION,
    'variables': {
        'bonete': 'bonete',
        'baygorria': 'baygorria',
        'palmar': 'palmar',
        'salto': 'salto',
        'hydros': ['bonete', 'baygorria', 'palmar', 'salto'],
        'PTigreA': 'PTigreA',
        'CTR': 'CTR',
        'APR': 'APR',
        'motores': 'motores',
        'Bio_desp': 'Bio_desp',
        'Bio_nodesp': 'Bio_nodesp',
        'thermal_new': 'thermal_new',
        'PTigreB': 'PTigreB',
        'thermals': ['PTigreA', 'CTR', 'APR', 'motores',
                     'Bio_desp', 'Bio_nodesp', 'thermal_new', 'PTigreB'],
        'wind': 'eoloDeci',
        'solar': 'solarDeci'
    },
    'current_value': True
}

RES_FILES_RESULTS = [PRODUCTION_BY_RESOURCE_TABLE, REVENUES_BY_RESOURCE_TABLE,
                     COSTS_BY_PARTICIPANT_TABLE, CAPACITIES]

multipliers = {'production': 1_000, 'costs': 1_000_000}


def current_values_from_res_table(option, sim_folder):
    """
    Process a table from a res file and return the current value of the variables
    selected in the option dictionary.
    """
    dataframe = process_from_res_file(option, sim_folder)
    results = {}

    key_table = option['key']

    for key_english, variables in option['variables'].items():
        key_result = key_table + '_' + key_english
        if check_columns(dataframe, variables):
            # get current value for each variables
            results[key_result] = get_current_value(dataframe, variables)
        else:
            results[key_result] = -17
    return results


def get_current_value(dataframe, variables):
    """
    Get the current value of the variables in the dataframe
    """
    # If variables is a string, convert it to a list
    if isinstance(variables, str):
        variables = [variables]

    # Sum the values of the specified variables in the second row (index 1)
    return dataframe[variables].iloc[1].sum()


def check_columns(df, column_name):
    # If column_name is a string, convert it to a list
    if isinstance(column_name, str):
        column_name = [column_name]

    # Check if all columns in the list are present in the DataFrame
    if all(name in df.columns for name in column_name):
        return True
    else:
        return False


def present_values_from_res_table(option, sim_folder):
    """
    Process a table from a res file and return the present value of the variables
    selected in the option dictionary.
    """
    dataframe = process_from_res_file(option, sim_folder)
    df_columns = dataframe.columns

    results = {}

    key_table = option['key']

    for key_interest, annual_interest in INTEREST_RATES.items():
        for key_english, variables in option['variables'].items():
            key_result = key_table + '_' + key_english + '_' + key_interest
            if check_columns(dataframe, variables):
                # get present value for each variables
                results[key_result] = proc.get_present_value(
                    dataframe, variables, annual_interest)
            else:
                results[key_result] = -17
    return results


def process_from_res_file(option, sim_folder):
    file_path = auxiliary.try_get_file(sim_folder, option['filename'])
    data_table = proc.read_xlt(file_path, option)
    # print(data_table)
    number_of_years = len(data_table[0]) - 1
    years = [2023 + y for y in range(number_of_years)]
    dataframe = pd.DataFrame({'year': years})
    for line in data_table:
        dataframe[line[0]] = line[1:]

    dataframe['datetime'] = pd.to_datetime(
        dataframe['year'].astype(str) + '-01-01 00:00:00')
    dataframe['datetime'] = dataframe['datetime'].dt.strftime(
        '%m/%d/%y %H:%M:%S')
    return dataframe
