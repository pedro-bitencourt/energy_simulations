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
import json
from time import sleep
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.run_module import Run
from src.run_processor_module import RunProcessor
from src.constants import DATETIME_FORMAT
from src.utils.logging_config import setup_logging


OUTPUT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/test_data/output')
RUN_FOLDER = Path('/Users/pedrobitencourt/energy_simulations/test_data/1.25_13041_1957_480')


class TestRun(unittest.TestCase):
    def setUp(self):
        # set up logging level
        setup_logging(level=logging.INFO)

        parent_folder = RUN_FOLDER.parent
        run_name: str = RUN_FOLDER.name
        # Break run name into capacities
        capacities = run_name.split('_')
        exogenous_capacity = float(capacities[0])
        wind_capacity = float(capacities[1])
        solar_capacity = float(capacities[2])
        thermal_capacity = float(capacities[3])
        general_parameters: dict = {'daily': True,
                            'name_subfolder': 'CAD-2024-DIARIA'}

        variables = {'hydro_factor': {'value': exogenous_capacity},
                     'wind': {'value': wind_capacity},
                     'solar': {'value': solar_capacity},
                     'thermal': {'value': thermal_capacity}}

        self.mock_run = Run(parent_folder, general_parameters, variables)
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

        #    def test_profits_data_dict(self):
        #        profits_data_dict = self.mock_run_processor.profits_data_dict()
        #        self.assertIsNotNone(profits_data_dict)
        #        print(f'{profits_data_dict=}')
        #        sleep(5)
        #
        #    def test_get_profits(self):
        #        profits: dict = self.mock_run_processor.get_profits()
        #        self.assertIsNotNone(profits)
        #        # Save to disk using json
        #        with open(OUTPUT_FOLDER / 'profits.json', 'w') as f:
        #            json.dump(profits, f)
        

    def test_construct_random_variables_df(self):
        random_variables = self.mock_run_processor.construct_random_variables_df(complete=True)
        # Save to disk
        random_variables.to_csv(OUTPUT_FOLDER / 'random_variables.csv', index=False)
        self.assertIsNotNone(random_variables)
        self.assertFalse(random_variables.isna().values.any(),
                         "DataFrame contains NaN values")
        parsed_dates = pd.to_datetime(
            random_variables['datetime'], format=DATETIME_FORMAT, errors='coerce')
        self.assertFalse(parsed_dates.isna().any(
        ), "DataFrame contains invalid dates in the 'datetime' column")
        print(f'{random_variables.head()=}')


if __name__ == '__main__':
    setup_logging(level=logging.INFO)
    unittest.main()
