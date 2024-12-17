# Description: This file contains the constants used in the project.
from pathlib import Path

from .utils.auxiliary import make_name



################################################################################################
# NAME FUNCTIONS
################################################################################################

def create_run_name(variables: dict):
    var_values: list[float] = [variable['value'] for variable in
                               variables.values()]
    name: str = make_name(var_values)
    return name


def create_investment_name(parent_name: str, exogenous_variables: dict):
    exog_var_values: list[float] = [variable['value'] for variable in
                                    exogenous_variables.values()]
    name: str = make_name(exog_var_values)
    name = f'{parent_name}_{name}'
    return name


################################################################################################
# PATHS
################################################################################################
BASE_PATH: Path = Path('/projects/p32342')
POSTE_FILEPATH = '/projects/p32342/aux/poste_dictionary.csv'

def initialize_paths_comparative_statics(base_path: str, name: str) -> dict:
    paths = {}
    # Folders
    paths['main'] = Path(f"{base_path}/raw/{name}")
    paths['results'] = Path(f"{base_path}/results/{name}")
    paths['random_variables'] = Path(
        f"{base_path}/results/{name}/random_variables")

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Files
    paths['bash'] = Path(
        f"{base_path}/raw/{name}/process.sh")
    paths['investment_results'] = paths['results'] / 'investment_results.csv'
    paths['conditional_means'] = paths['results'] / 'conditional_means.csv'
    return paths

def initialize_paths_investment_problem(folder: Path, name: str) -> dict:
    paths: dict[str, Path] = {}
    # Folders
    paths['parent_folder'] = folder
    paths['folder'] = folder / name

    # Files
    paths['optimization_trajectory'] = folder / name /\
        f'{name}_trajectory.json'
    paths['bash'] = folder / name / f'{name}.sh'

    paths['slurm_out'] = folder / f'{name}.out'
    paths['slurm_err'] = folder / f'{name}.err'

    # create the directory
    paths['folder'].mkdir(parents=True, exist_ok=True)
    return paths


def initialize_paths_run(parent_folder: Path, name: str, subfolder: str) -> dict:
    """
    Initialize a dictionary with relevant paths for the run.
    """
    def format_windows_path(path_str):
        path_str = str(path_str)
        # Replace forward slashes with backslashes
        windows_path = path_str.replace('/', '\\')
        # Add Z: at the start of the path
        windows_path = 'Z:' + windows_path
        # If the path doesn't end with a backslash, add one
        if not windows_path.endswith('\\'):
            windows_path += '\\'
        return windows_path

    paths = {}
    paths['parent_folder'] = parent_folder
    folder = parent_folder / name
    paths['folder'] = folder
    # Convert the output path to a Windows path, for use in the .xml file
    paths['folder_windows'] = format_windows_path(paths['folder'])

    paths['subfolder'] = folder / subfolder

    paths['slurm_out'] = folder / f'{name}.out'
    paths['slurm_err'] = folder / f'{name}.err'

    # Add paths for results and price distribution files
    paths['random_variables'] = paths['folder'] / \
        'random_variables.csv'
    return paths


################################################################################################
# PARSING CONFIGURATIONS
################################################################################################
# Extraction configurations
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
        'numeric_columns': [f'{i}' for i in range(0, 114)],
        'keep_columns': ['datetime'] + [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': False,
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
        'numeric_columns': [f'{i}' for i in range(0, 114)],
        'keep_columns': ['datetime'] + [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': True,
}

VARIABLE_COSTS_THERMAL_DF = {
    'name': 'variable_costs',
    'filename': 'TER_thermal/costos*xlt',
    'table_pattern': {
        'start': 'CANT_POSTE',
        'end': None
    },
    'columns_options': {
        'drop_columns': ['PROMEDIO_ESC', 'poste'],
        'rename_columns': {
            **{f'ESC{i}': f'{i}' for i in range(0, 114)},
            '': 'datetime'
        },
        'numeric_columns': [f'{i}' for i in range(0, 114)],
    },
    'delete_first_row': True,
    'convert_poste': False
}

SALTO_WATER_LEVEL_DF = {
    'name': 'salto_water_level',
    'filename': 'HID_salto/cota*xlt',
    'table_pattern': {
        'start': 'CANT_POSTE',
        'end': None
    },
    'columns_options': {
        'drop_columns': ['PROMEDIO_ESC', 'poste'],
        'rename_columns': {
            **{f'ESC{i}': f'{i}' for i in range(0, 114)},
            '': 'datetime'
        },
        'numeric_columns': [f'{i}' for i in range(0, 114)],
        'keep_columns': ['datetime'] + [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': True,
    'convert_poste': False
}

# constants for parsing the results
DATETIME_FORMAT = '%m/%d/%y %H:%M'
SCENARIOS = [f'{i}' for i in range(0, 114)]

# Error codes
UNSUCCESSFUL_RUN: int = 13
ERROR_CODE_UNSUCCESSFUL_ITERATION: int = 14
