'''
Unit tests for the run_module.py and run_processor_module.py

Currently tests:
- Extracting the marginal costs from the output files
- Calculating the profits from the output files
- Creating a Run object with the correct attributes
- Creating a RunProcessor object with the correct attributes
'''
import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.run_module import Run
from src.run_processor_module import RunProcessor
from src.constants import DATETIME_FORMAT, SCENARIOS


COLUMNS_TO_CHECK = ['datetime'] + SCENARIOS


PARENT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data/comparative_statics/salto_capacity_v3/salto_capacity_v3_investment_101.25')
OUTPUT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data/output')

# Not necessary
REQUESTED_TIME_RUN: float = 6.5
REQUESTED_TIME_SOLVER: float = 16

class TestRun(unittest.TestCase):
    def setUp(self):
        # Set up the mock Run object
        name = 'salto_capacity_v3'
        exogenous_capacity = 101.25
        wind_capacity = 4540
        solar_capacity = 1769
        thermal_capacity = 1051
        general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA',
                            'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                            'email': 'pedro.bitencourt@u.northwestern.edu',
                            'annual_interest_rate': 0.0,
                            'years_run': 6.61,
                            'requested_time_run': REQUESTED_TIME_RUN,
                            'requested_time_solver': REQUESTED_TIME_SOLVER}

        variables = {'hydro_factor': {'pattern': 'HYDRO_FACTOR',
                                      'label': 'Hydro Factor',
                                      'value': exogenous_capacity},
                     'wind': {'pattern': 'WIND_CAPACITY',
                              'label': 'Wind Capacity',
                              'value': wind_capacity},
                     'solar': {'pattern': 'SOLAR_CAPACITY',
                               'label': 'Solar Capacity',
                               'value': solar_capacity},
                     'thermal': {'pattern': 'THERMAL_CAPACITY',
                                 'label': 'Thermal Capacity',
                                 'value': thermal_capacity}}

        self.mock_run = Run(PARENT_FOLDER, general_parameters, variables)
        # Initialize the RunProcessor object
        self.mock_run_processor = RunProcessor(self.mock_run)

    def tearDown(self):
        pass
        # delete all .csv files in output folder
        #for file in OUTPUT_FOLDER.glob('*.csv'):
        #    file.unlink(missing_ok=True)


#    def test__extract_marginal_cost(self):
#        dataframe = self.mock_run_processor._extract_marginal_costs_df()
#        self.assertIsNotNone(dataframe)
#        self.assertFalse(dataframe.isna().values.any(),
#                         "DataFrame contains NaN values")
#        parsed_dates = pd.to_datetime(
#            dataframe['datetime'], format=DATETIME_FORMAT, errors='coerce')
#        self.assertFalse(parsed_dates.isna().any(
#        ), "DataFrame contains invalid dates in the 'datetime' column")
#        print(f'{dataframe.head()=}')
#        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)
#
#    def test_production_participant(self):
#        participant_key = 'thermal'
#        dataframe = self.mock_run_processor.production_participant(participant_key)
#        self.assertIsNotNone(dataframe)
#        print(f'{dataframe.head()=}')
#        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)
#
#    def test_variable_costs_participant(self):
#        participant_key = 'thermal'
#        dataframe = self.mock_run_processor.variable_costs_participant(participant_key)
#        self.assertIsNotNone(dataframe)
#        print(f'{dataframe.head()=}')
#        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)

    def test_get_profits(self):
        profits = self.mock_run_processor.get_profits()
        self.assertIsNotNone(profits)


    def test_get_random_variables_df(self):
        random_variables = self.mock_run_processor.get_random_variables_df(lazy=False, complete=True)
        # Save to disk
        random_variables.to_csv(OUTPUT_FOLDER / 'random_variables.csv', index=False)
        self.assertIsNotNone(random_variables)
        self.assertFalse(random_variables.isna().values.any(),
                         "DataFrame contains NaN values")
        parsed_dates = pd.to_datetime(
            random_variables['datetime'], format=DATETIME_FORMAT, errors='coerce')
        self.assertFalse(parsed_dates.isna().any(
        ), "DataFrame contains invalid dates in the 'datetime' column")
        self.assertEqual(random_variables.columns.tolist(), COLUMNS_TO_CHECK)

if __name__ == '__main__':
    unittest.main()
