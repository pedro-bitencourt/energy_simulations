from pathlib import Path

LOCAL: bool = False

# slurm configurations
HOURS_REQUEST_RUN: int = 2
MINUTES_REQUEST_RUN: int = 30

# constants
DATETIME_FORMAT = '%m/%d/%y %H:%M'
SCENARIOS = [f'{i}' for i in range(0, 114)]

# economic data
ANNUAL_INTEREST_RATE = 0.05
COSTS = {
    'wind': {'oem': 20_000, 'installation': 1_300_000},
    'solar': {'oem': 7_300, 'installation': 1_160_000}
}

# paths
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

MARGINAL_COST_DF_CONFIG = {
    'name': 'marginal_cost',
    'filename': r'cosmar/cosmar_esc_im.xlt',
    'folder_key': 'sim',
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
