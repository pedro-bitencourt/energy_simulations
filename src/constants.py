# Description: This file contains the constants used in the project.
from pathlib import Path

# constants for parsing the results
DATETIME_FORMAT = '%m/%d/%y %H:%M'
SCENARIOS = [f'{i}' for i in range(0, 114)]

# costs are in USD per MW for installation and USD per MW per year for O&M
COSTS = {
    'wind': {'oem': 20_000, 'installation': 1_300_000, 'lifetime': 25},
    'solar': {'oem': 7_300, 'installation': 1_160_000, 'lifetime': 35},
    'thermal': {'oem': 12_000, 'installation': 9_750_000, 'lifetime': 20}
}

# Error codes
UNSUCCESSFUL_RUN: int = 2
ERROR_CODE_UNSUCCESSFUL_ITERATION: int = 3

# Optimization constants
# Maximum number of iterations for the optimization algorithm
MAX_ITER: int = 15
# Delta is used to calculate the numerical derivatives of the profits
DELTA: float = 10
# Threshold for the profits to be considered converged, in percentage of the
# installation cost
THRESHOLD_PROFITS: float = 0.01  # Default threshold for convergence

# Relevant paths
BASE_PATH: Path = Path('/projects/p32342')
POSTE_FILEPATH = '/projects/p32342/aux/poste_dictionary.csv'

# Plotting configurations
HEATMAP_CONFIGS = [
    {'variables': {'x': {'key': 'hydro_factor', 'label': 'Hydro Factor'},
                   'y': {'key': 'thermal_capacity', 'label': 'Thermal Capacity'},
                   'z': {'key': 'price_avg', 'label': 'Profit'}},
     'filename': 'price_avg_heatmap.png',
     'title': 'Price Heatmap'}
]

ONE_D_PLOTS_CONFIGS = [
    # Optimal capacities of wind and solar
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'wind', 'label': 'Optimal Wind Capacity'},
            {'key': 'solar', 'label': 'Optimal Solar Capacity'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Capacity (MW)'},
        'title': 'Optimal Wind and Solar Capacities',
        'filename': 'wind_solar_capacities.png'
    },
    # Total production by resources
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'total_production_hydros', 'label': 'Hydro'},
            {'key': 'total_production_thermals', 'label': 'Thermal'},
            {'key': 'total_production_combined_cycle',
                'label': 'Combined Cycle'},
            {'key': 'total_production_wind', 'label': 'Wind'},
            {'key': 'total_production_solar', 'label': 'Solar'},
            {'key': 'total_production_import_export',
                'label': 'Import/Export'},
            {'key': 'total_production_demand', 'label': 'Demand'},
            {'key': 'total_production_blackout', 'label': 'Blackout'},

        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Total Production (GWh)'},
        'title': 'Total Production by Resources',
        'filename': 'total_production.png'
    },
    # New thermal production
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'total_production_hydros', 'label': 'Hydro'},
            {'key': 'new_thermal_production', 'label': 'New Thermal'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Total Production (GWh)'},
        'title': 'Production of Hydros and New Thermal',
        'filename': 'total_production_new_thermal.png'
    },
    # Average price
    {
        'x_key': 'hydro_factor',
        'y_variables': [
            {'key': 'price_avg', 'label': 'Average Price'}
        ],
        'axis_labels': {'x': 'Hydro Factor', 'y': 'Price ($/MWh)'},
        'title': 'Average Price',
        'filename': 'average_price.png'
    }
]


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
