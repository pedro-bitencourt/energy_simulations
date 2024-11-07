'''
Unit tests for the run_module.py and run_processor_module.py

Currently tests:
- Extracting the marginal costs from the output files
- Calculating the profits from the output files
- Creating a Run object with the correct attributes
- Creating a RunProcessor object with the correct attributes
'''
import unittest
from pathlib import Path
import pandas as pd
from run_module import Run
from run_processor_module import RunProcessor

from constants import DATETIME_FORMAT


MOCK_OUTPUT_FOLDER = Path(
'/Users/pedrobitencourt/quest/data/renewables/zero_mc_thermal/10_45_0_180')
MOCK_INPUT_FOLDER = Path('~')


class TestRun(unittest.TestCase):
    def setUp(self):

        self.mock_input_folder = MOCK_INPUT_FOLDER
        self.mock_output_folder = MOCK_OUTPUT_FOLDER

        def run_name_function(variables):
            hydro_key: int = int(10*variables['hydro']['value'])
            thermal_key: int = int(variables['thermal']['value'])
            wind_key: int = int(variables['wind']['value'])
            solar_key: int = int(variables['solar']['value'])
            name: str = f"{hydro_key}_{thermal_key}_{wind_key}_{solar_key}"
            return name

        self.mock_general_parameters = {'daily': True,
                                        'name_subfolder': 'CAD-2024-DIARIA',
                                        'name_function': run_name_function}

        self.mock_variables = {'wind': {'value': 0, 'pattern': 'WIND_CAPACITY'},
                               'solar': {'value': 180, 'pattern': 'SOLAR_CAPACITY'}}
        self.mock_run = Run(self.mock_input_folder,
                            self.mock_general_parameters,
                            self.mock_variables,
                            output_folder=self.mock_output_folder)
        self.mock_run_processor = RunProcessor(self.mock_run)

    def tearDown(self):
        # delete all .csv files in output folder
        for file in self.mock_output_folder.glob('*.csv'):
            file.unlink(missing_ok=True)
        pass

    def test_get_price_distribution(self):
        price_distribution = self.mock_run_processor.get_price_distribution()
        self.assertIsNotNone(price_distribution)

    def test_get_price_results(self):
        price_results: dict = self.mock_run_processor.get_price_results()
        for key, value in price_results.items():
            self.assertIsNotNone(value)

    def test_extract_marginal_cost(self):
        dataframe = self.mock_run_processor.extract_marginal_costs_df()
        self.assertIsNotNone(dataframe)
        self.assertFalse(dataframe.isna().values.any(),
                         "DataFrame contains NaN values")
        parsed_dates = pd.to_datetime(
            dataframe['datetime'], format=DATETIME_FORMAT, errors='coerce')
        self.assertFalse(parsed_dates.isna().any(
        ), "DataFrame contains invalid dates in the 'datetime' column")

    def test_get_profits(self):
        profits = self.mock_run_processor.get_profits()
        print(profits)
        self.assertIsNotNone(profits)


if __name__ == '__main__':
    unittest.main()
