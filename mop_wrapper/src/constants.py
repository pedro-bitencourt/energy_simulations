"""
Filename: constants.py

This file contains constants such as Slurm default values, paths, and configurations for parsing 
the raw data from MOP.
"""

from pathlib import Path
from .utils.auxiliary import make_name

################################################################################################
# SLURM DEFAULT CONFIGURATIONS
# Moved to slurm_utils.py

################################################################################################
# NAME FUNCTIONS
################################################################################################


def create_run_name(variables: dict) -> str:
    # Order the variables alphabetically
    variables: dict = dict(sorted(variables.items()))
    # Extract the values directly, assuming they are floats
    var_values: list[float] = list(variables.values())
    # Assuming make_name processes the list correctly
    name: str = make_name(var_values)
    return name


def create_solver_name(parent_name: str, exogenous_variables: dict) -> str:
    exog_var_values: list[float] = [variable['value'] for variable in
                                    exogenous_variables.values()]
    print(exog_var_values)
    name: str = make_name(exog_var_values)
    name = f'{parent_name}_{name}'
    return name


################################################################################################
# PATHS
################################################################################################

BASE_PATH_TEMP: Path = Path(__file__).resolve().parent.parent.parent.parent
BASE_PATH: Path = Path(BASE_PATH_TEMP.as_posix().removeprefix("/gpfs"))


CONFIG_PATH: Path = BASE_PATH / 'code/config/config.yaml'


def initialize_paths_comparative_statics(base_path: str, name: str) -> dict:
    paths = {}
    base_path: Path = Path(base_path)
    paths['main'] = base_path / "sim" / name
    paths['temp'] = paths['main'] / 'temp'
    paths['results'] = paths['main'] / 'results'
    paths['trajectories'] = paths['main'] / 'trajectories'
    paths['raw'] = paths['main'] / "raw"

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    files_dict: dict[str, str] = {
        'bash': 'temp/process.sh',
        'slurm_out': 'temp/{name}.out',
        'slurm_err': 'temp/{name}.err',
        'solver_results': 'results/solver_results.csv',
        'profits': 'results/profits.csv',
        'conditional_means': 'results/conditional_means.csv'
    }
    paths.update({key: paths['main'] / value for key,
                 value in files_dict.items()})

    return paths


def initialize_paths_solver(parent_folder: Path, name: str) -> dict:
    paths: dict[str, Path] = {}
    # Folders
    paths['parent_folder'] = parent_folder
    paths['folder'] = parent_folder / name

    # Subfolders
    paths['temp'] = paths['folder'] / 'temp'
    paths['results'] = paths['folder'] / 'results'

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # Files
    files_folder_dict: dict[str, str] = {
        'bash': f'temp/{name}.sh',
        'slurm_out': f'temp/{name}.out',
        'slurm_err': f'temp/{name}.err',
        'solver_results': 'results/solver_results.json'
    }
    paths.update({key: paths['folder'] / value for key,
                 value in files_folder_dict.items()})
    files_parent_folder_dict: dict[str, str] = {
        'solver_trajectory': f'trajectories/{name}_trajectory.json',
        'raw': f'raw/{name}.csv'
    }
    paths.update({key: parent_folder / value for key,
                 value in files_parent_folder_dict.items()})
    return paths


def initialize_paths_run(parent_folder: Path, name: str, subfolder: str) -> dict:
    """
    Initialize a dictionary with relevant paths for the run.
    """
    def format_windows_path(path_str):
        path_str = str(path_str)
        windows_path = path_str.replace('/', '\\')
        windows_path = 'Z:' + windows_path
        if not windows_path.endswith('\\'):
            windows_path += '\\'
        return windows_path

    paths = {}
    paths['parent_folder'] = parent_folder
    folder = parent_folder / name
    paths['folder'] = folder
    paths['xml'] = folder / f"{name}.xml"
    # Convert the output path to a Windows path, for use in the .xml file
    paths['folder_windows'] = format_windows_path(paths['folder'])

    paths['subfolder'] = folder / subfolder

    paths['slurm_out'] = folder / f'{name}.out'
    paths['slurm_err'] = folder / f'{name}.err'

    # Add paths for results and price distribution files
    paths['random_variables'] = paths['folder'] / 'random_variables.csv'
    return paths


################################################################################################
# PARSING CONFIGURATIONS
################################################################################################
# Extraction configurations, used to parse the raw data from MOP.
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
    'filename': 'DEM_demand/potencias*xlt',
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

WATER_LEVEL_DF = {
    'name': 'salto_water_level',
    'filename': '',
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
