import pandas as pd
from datetime import datetime, timedelta

# GLOBAL VARIABLES
SCENARIOS = [f'{i}' for i in range(0, 114)]
DATETIME_FORMAT = '%m/%d/%y %H:%M'
POSTE_FILEPATH = '/projects/p32342/code/aux/poste_dictionary.csv'
PARTICIPANTS = ['wind', 'solar']  # ,thermal]
ANNUAL_INTEREST_RATE = 0.10

###############################################################################
# PATTERNS CONFIGURATIONS
# These configurations are used to extract data from the output files.
# Each pattern has the following keys
# - pattern: Regular expression to match the line
# - variable_name: Name of the variable to store the result
# - paths_key: Key of the paths dictionary where the file is located
# - complement: Complement of the file name
###############################################################################
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

PATTERNS = pd.DataFrame(PATTERNS_LIST)

##############################################################################
# DATAFRAMES TO SAVE
##############################################################################

# used in the participant_module
DATAFRAMES_TO_SAVE = []


def process_marginal_cost(marginal_cost_df):
    def datetime_from_row_marginal_cost(row):
        """
        Constructs a datetime column from a row.
        """
        start_date = pd.to_datetime(row['FechaInicialPaso'], format='%Y/%m/%d')
        hours_added = pd.to_timedelta(pd.to_numeric(
            row['Int.MuestreoDelPaso'], errors='coerce'),
            unit='h')
        return (start_date + hours_added).strftime(DATETIME_FORMAT)
    marginal_cost_df = pd.DataFrame(
        data=marginal_cost_df[1:], columns=marginal_cost_df[0])
    marginal_cost_df['datetime'] = marginal_cost_df.apply(datetime_from_row_marginal_cost,
                                                          axis=1)
    return marginal_cost_df


DATAFRAMES_TO_SAVE.append({
    'name': 'marginal_cost',
    'filename': r'cosmar/cosmar_esc_im.xlt',
    'folder_key': 'sim',
    'table_pattern': {
        'start': 'FechaInicialPaso',
        'end': None
    },
    'columns_options': {
        'drop_columns':
        ['FechaInicialPaso', 'Int.MuestreoDelPaso', 'PROMEDIO', ''],
        'rename_columns': None,
        'numeric_columns': [f'{i}' for i in range(0, 114)]
    },
    'delete_first_row': False,
    'output_filename': 'marginal_cost',
    'process': process_marginal_cost
})

###############################################################################
# RESULTS FROM DATAFRAMES
###############################################################################
RESULTS_FROM_DATAFRAMES = []


def results_from_marginal_cost(paths):
    results = {}
    file_path = paths['marginal_cost']
    mc_df = pd.read_csv(file_path)
    try:
        results['avg_hours_blackout'] = mc_df.applymap(
            lambda x: x == 4000).sum().sum()/len(SCENARIOS)
        results['avg_price_simple'] = mc_df.mean().mean()
        return results
    except Exception as e:
        print(f'Error: {e}')
        return False


# RESULTS_FROM_DATAFRAMES.append(results_from_marginal_cost)


# def average_price_weighted(paths):
#    mc_file_path = paths['marginal_cost']
#    mc_df = pd.read_csv(mc_file_path)
#    mc_df = mc_df[SCENARIOS]
#    demand_file_path = paths['demand']
#    demand_df = pd.read_csv(demand_file_path)
#    demand_df = demand_df[SCENARIOS]
#    weighted_values = mc_df * demand_df
#    sum_weights = mc_df.sum().sum()
#    weighted_avg = weighted_values.sum().sum()/sum_weights
#    return weighted_avg


###############################################################################
# FINAL RESULTS MANIPULATIONS
###############################################################################
FINAL_RESULTS_OPERATIONS = {}

FINAL_RESULTS_OPERATIONS['optimization_time_h'] = lambda df: df[
    'optimization_time'] / 3600
FINAL_RESULTS_OPERATIONS['simulation_time_h'] = lambda df: df['simulation_time'
                                                              ] / 3600
FINAL_RESULTS_OPERATIONS['total_time_h'] = lambda df: (df[
    'simulation_time'] + df['optimization_time']) / 3600

FINAL_RESULTS_OPERATIONS['avg_price_thermal'] = lambda df: (
    df['all_thermal_revenues']/df['all_thermal_production']
)
