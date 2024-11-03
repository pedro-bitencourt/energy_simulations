# Description: This file contains the constants used in the project.
from pathlib import Path
import platform

# auto checks whether we're on quest or on a local mac machine
LOCAL: bool = platform.system() == 'Darwin'

# slurm configurations
HOURS_REQUEST_RUN: int = 2
MINUTES_REQUEST_RUN: int = 30

REQUESTED_TIME_RUN = f'{HOURS_REQUEST_RUN}:{MINUTES_REQUEST_RUN}:00'

# constants for parsing the results
DATETIME_FORMAT = '%m/%d/%y %H:%M'
SCENARIOS = [f'{i}' for i in range(0, 114)]

# economic data
ANNUAL_INTEREST_RATE = 0.00
# costs are in USD per MW for installation and USD per MW per year for O&M
COSTS = {
    'wind': {'oem': 20_000, 'installation': 1_300_000, 'lifetime': 25},
    'solar': {'oem': 7_300, 'installation': 1_160_000, 'lifetime': 35},
    'thermal': {'oem': 12_000, 'installation': 9_750_000, 'lifetime': 20}
}

# relevant paths
if LOCAL:
    BASE_PATH: Path = Path('/Users/pedrobitencourt/quest')
    POSTE_FILEPATH = '/Users/pedrobitencourt/quest/data/renewables/poste_dictionary.csv'
else:
    BASE_PATH: Path = Path('/projects/p32342')
    POSTE_FILEPATH = '/projects/p32342/aux/poste_dictionary.csv'

# extraction configurations
PATTERNS_LIST = [
    {
        'pattern': r'TOTAL OPTIMIZAR es de: (\d+) seg',
        'variable_name': 'optimization_time',
        'paths_key': 'opt',
        'complement': 'tiempos_optimizacion.txt'
    }, {
        'pattern': r'TotalSimulacion es de: (\d+) seg',
        'variable_name': 'simulation_time',
        'paths_key': 'sim',
        'complement': 'tiempos_optimizacion.txt'
    }, {
        'pattern': "EXCEPTION_ACCESS_VIOLATION",
        'variable_name': 'exception_access',
        'paths_key': 'slurm_output',
        'complement': None
    }, {
        'pattern': "TERMINÓ SIMULACIÓN",
        'variable_name': 'successful',
        'paths_key': 'slurm_output',
        'complement': None
    }, {
        'pattern': "DUE TO TIME LIMIT",
        'variable_name': 'time_limit_reached',
        'paths_key': 'slurm_output',
        'complement': None
    }
    #
    # {
    #    'pattern': None,
    #    'variable_name': None,
    #    'paths_key': None,
    #    'complement': None
    # }
]

MARGINAL_COST_DF = {
    'name': 'marginal_cost',
    'filename': r'cosmar/cosmar_esc_im.xlt',
    'table_pattern': {
        'start': 'FechaInicialPaso',
        'end': None
    },
    'columns_options': {
        'drop_columns':
        ['PROMEDIO', ''],
        'rename_columns': None,
        'numeric_columns': [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': False,
    'output_filename': 'marginal_cost'
}

DEMAND_DF = {
    'name': 'demand',
    'filename': 'DEM_demandaPrueba/potencias*xlt',
    'table_pattern': {
        'start': 'CANT_POSTE',
        'end': None
    },
    'columns_options': {
        'drop_columns': ['PROMEDIO_ESC'],
        'rename_columns': {
            **{f'ESC{i}': f'{i}' for i in range(0, 114)},
            '': 'paso_start'
        },
        'numeric_columns': [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': True,
    'output_filename': 'demand'
}

################################################################################################
# RESUMEN MODULE
# REVENUES_BY_RESOURCE_TABLE = {
#    'key': 'revenues_discounted',
#    'filename': r'resUnico*.xlt',
#    'table_pattern': {
#        'start':  'VALOR MEDIO AL COSTO MARGINAL SIN TOPE(USD/MWh)',
#        'end': 'RESUMEN DE HORAS DE FALLA',
#    },
#    **BASIC_RES_OPTION,
#    'variables': {
#        'hydros': ['bonete', 'baygorria', 'palmar', 'salto'],
#        'thermals': ['PTigreA', 'CTR', 'APR', 'motores',
#                     'Bio_desp', 'Bio_nodesp', 'thermal_new', 'PTigreB'],
#        'wind': 'eoloDeci',
#        'solar': 'solarDeci',
#        'demand': 'demandaPrueba',
#        'import_export': ['impoEstacArg', 'impoBraRiv', 'impoBrMelo',
#                          'excedentes', 'sumidero', 'expLago'],
#        'blackout': ['demandaPrueba_EscFalla0', 'demandaPrueba_EscFalla1',
#                     'demandaPrueba_EscFalla2', 'demandaPrueba_EscFalla3'],
#        'bonete': 'bonete',
#        'baygorria': 'baygorria',
#        'palmar': 'palmar',
#        'salto': 'salto',
#        'PTigreA': 'PTigreA',
#        'CTR': 'CTR',
#        'APR': 'APR',
#        'motores': 'motores',
#        'Bio_desp': 'Bio_desp',
#        'Bio_nodesp': 'Bio_nodesp',
#        'thermal_new': 'thermal_new',
#        'PTigreB': 'PTigreB',
#        'impoEstacArg': 'impoEstacArg',
#        'impoBraRiv': 'impoBraRiv',
#        'impoBrMelo': 'impoBrMelo',
#        'excedentes': 'excedentes',
#        'sumidero': 'sumidero',
#        'expLago': 'expLago',
#        'demandaPrueba_EscFalla0': 'demandaPrueba_EscFalla0',
#        'demandaPrueba_EscFalla1': 'demandaPrueba_EscFalla1',
#        'demandaPrueba_EscFalla2': 'demandaPrueba_EscFalla2',
#        'demandaPrueba_EscFalla3': 'demandaPrueba_EscFalla3',
#    }
# }
#
# COSTS_BY_PARTICIPANT_TABLE = {
#   'key': 'costs',
#   'filename': r'resUnico*.xlt',
#   'table_pattern': {
#       'start': 'COSTOS ESPERADOS EN MUSD ',
#       'end': 'COSTO DE IMPACTOS Y DE CONTRATOS DE ENERGIA'
#   },
#   **BASIC_RES_OPTION,
#   'variables': {
#       'thermal': ['PTigreA', 'CTR', 'APR', 'motores', 'Bio_desp',
#                   'Bio_nodesp', 'thermal_new', 'PTigreB'],
#       'PTigreA': 'PTigreA',
#       'CTR': 'CTR',
#       'APR': 'APR',
#       'motores': 'motores',
#       'Bio_desp': 'Bio_desp',
#       'Bio_nodesp': 'Bio_nodesp',
#       'thermal_new': 'thermal_new',
#       'PTigreB': 'PTigreB',
#       'impoEstacArg': 'impoEstacArg',
#       'impoBraRiv': 'impoBraRiv',
#       'impoBrMelo': 'impoBrMelo',
#       'excedentes': 'excedentes',
#       'sumidero': 'sumidero',
#       'expLago': 'expLago',
#       'demandaPrueba_EscFalla0': 'demandaPrueba_EscFalla0',
#       'demandaPrueba_EscFalla1': 'demandaPrueba_EscFalla1',
#       'demandaPrueba_EscFalla2': 'demandaPrueba_EscFalla2',
#       'demandaPrueba_EscFalla3': 'demandaPrueba_EscFalla3'
#   }
# }
#
# CAPACITIES = {
#    'key': 'capacities',
#    'filename': r'resUnico*.xlt',
#    'table_pattern': {
#        'start': 'POTENCIA TOTAL DE GENERADORES(MW)',
#        'end': 'ENERGIAS ESPERADAS POR TIPO DE RECURSO EN GWh'
#    },
#    **BASIC_RES_OPTION,
#    'variables': {
#        'bonete': 'bonete',
#        'baygorria': 'baygorria',
#        'palmar': 'palmar',
#        'salto': 'salto',
#        'hydros': ['bonete', 'baygorria', 'palmar', 'salto'],
#        'PTigreA': 'PTigreA',
#        'CTR': 'CTR',
#        'APR': 'APR',
#        'motores': 'motores',
#        'Bio_desp': 'Bio_desp',
#        'Bio_nodesp': 'Bio_nodesp',
#        'thermal_new': 'thermal_new',
#        'PTigreB': 'PTigreB',
#        'thermals': ['PTigreA', 'CTR', 'APR', 'motores',
#                     'Bio_desp', 'Bio_nodesp', 'thermal_new', 'PTigreB'],
#        'wind': 'eoloDeci',
#        'solar': 'solarDeci'
#    },
#    'current_value': True
# }
#
# RES_FILES_RESULTS = [PRODUCTION_BY_RESOURCE_TABLE, REVENUES_BY_RESOURCE_TABLE,
#                     COSTS_BY_PARTICIPANT_TABLE, CAPACITIES]
