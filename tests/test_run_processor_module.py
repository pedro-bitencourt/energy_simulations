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


PARENT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data')
OUTPUT_FOLDER: Path = PARENT_FOLDER / 'output'

# Not necessary
REQUESTED_TIME_RUN: int = 6.5
REQUESTED_TIME_SOLVER: int = 16

class TestRun(unittest.TestCase):
    def setUp(self):
        # Set up the mock Run object
        name = 'salto_capacity'
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
                                      'value': 1},
                     'wind': {'pattern': 'WIND_CAPACITY',
                              'label': 'Wind Capacity',
                              'value': 3295},
                     'solar': {'pattern': 'SOLAR_CAPACITY',
                               'label': 'Solar Capacity',
                               'value': 1670},
                     'thermal': {'pattern': 'THERMAL_CAPACITY',
                                 'label': 'Thermal Capacity',
                                 'value': 107}}

        self.mock_run = Run(PARENT_FOLDER, general_parameters, variables)
        # Initialize the RunProcessor object
        self.mock_run_processor = RunProcessor(self.mock_run)

    def tearDown(self):
        # delete all .csv files in output folder
        for file in OUTPUT_FOLDER.glob('*.csv'):
            file.unlink(missing_ok=True)


    def test__extract_marginal_cost(self):
        dataframe = self.mock_run_processor._extract_marginal_costs_df()
        self.assertIsNotNone(dataframe)
        self.assertFalse(dataframe.isna().values.any(),
                         "DataFrame contains NaN values")
        parsed_dates = pd.to_datetime(
            dataframe['datetime'], format=DATETIME_FORMAT, errors='coerce')
        self.assertFalse(parsed_dates.isna().any(
        ), "DataFrame contains invalid dates in the 'datetime' column")
        print(f'{dataframe.head()=}')
        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)


    def test_production_participant(self):
        participant_key = 'thermal'
        dataframe = self.mock_run_processor.production_participant(participant_key)
        self.assertIsNotNone(dataframe)
        print(f'{dataframe.head()=}')
        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)

    def test_variable_costs_participant(self):
        participant_key = 'thermal'
        dataframe = self.mock_run_processor.variable_costs_participant(participant_key)
        self.assertIsNotNone(dataframe)
        print(f'{dataframe.head()=}')
        self.assertEqual(dataframe.columns.tolist(), COLUMNS_TO_CHECK)

if __name__ == '__main__':
    unittest.main()
