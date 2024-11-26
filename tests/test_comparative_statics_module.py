import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.comparative_statics_module import ComparativeStatics
from src.constants import DATETIME_FORMAT, SCENARIOS

import logging

# 1. Basic level setting
logging.basicConfig(level=logging.INFO)  # Set global level
logger = logging.getLogger(__name__)

OUTPUT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data/output')
PARENT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data')

REQUESTED_TIME_SOLVER = 1
REQUESTED_TIME_RUN = 1

COLUMNS_TO_CHECK = ['run','datetime'] + SCENARIOS


class TestComparativeStatics(unittest.TestCase):
    def setUp(self):
        name = 'salto_capacity'
        general_parameters: dict = {'daily': True,
                                    'name_subfolder': 'CAD-2024-DIARIA',
                                    'xml_basefile': f'/projects/p32342/code/xml/{name}.xml',
                                    'email': 'pedro.bitencourt@u.northwestern.edu',
                                    'annual_interest_rate': 0.0,
                                    'years_run': 6.61,
                                    'requested_time_run': REQUESTED_TIME_RUN,
                                    'requested_time_solver': REQUESTED_TIME_SOLVER}
        exogenous_variable_name: str = 'hydro_factor'
        exogenous_variable_pattern: str = 'HYDRO_FACTOR'
        exogenous_variable_label: str = 'Hydro Factor'
        exogenous_variable_grid: list[float] = [1, 2]
        
        exogenous_variables: dict[str, dict] = {
            exogenous_variable_name: {'pattern': exogenous_variable_pattern,
                                      'label': exogenous_variable_label},
        }
        endogenous_variables: dict[str, dict] = {
            'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
            'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
            'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
        }
        exogenous_variables_grid: dict[str, np.ndarray] = {
            exogenous_variable_name: np.array(exogenous_variable_grid)}
        variables: dict[str, dict] = {'exogenous': exogenous_variables,
                                      'endogenous': endogenous_variables}
        
        # Create the ComparativeStatics object
        self.mock_comparative_statics = ComparativeStatics(name,
                                                 variables,
                                                 exogenous_variables_grid,
                                                 general_parameters,
                                                    PARENT_FOLDER)
        
    def tearDown(self):
        # delete all .csv files in output folder
        #for file in OUTPUT_FOLDER.glob('*.csv'):
        #    file.unlink(missing_ok=True)
        pass

    def test_construct_new_results(self): 
        # Test the construction of new results
        # Construct the new results
        new_results = self.mock_comparative_statics.construct_new_results()
        # Check the results
        self.assertIsInstance(new_results, dict)
        print(f'{new_results=}')


#    def test_get_dataframe(self):
#        def assert_dataframe(df: pd.DataFrame):
#            self.assertIsInstance(df, pd.DataFrame)
#            self.assertFalse(df.empty)
#            # Check the columns
#            self.assertEqual(set(df.columns.tolist()), set(COLUMNS_TO_CHECK))
#            # Check the datetime format
#            self.assertEqual(df['datetime'].dtype, 'datetime64[ns]')
#            # Check for NaN values
#            self.assertFalse(df.isna().values.any(), "DataFrame contains NaN values")
#            # Check for invalid dates
#            parsed_dates = pd.to_datetime(
#                df['datetime'], format=DATETIME_FORMAT, errors='coerce')
#            self.assertFalse(parsed_dates.isna().any(),
#                             "DataFrame contains invalid dates in the 'datetime' column")
#
#        # Test marginal cost dataframe
#        marginal_cost_df = self.mock_comparative_statics.get_dataframe('marginal_cost')
#        assert_dataframe(marginal_cost_df)
#        # Save the dataframe to a file
#        marginal_cost_df.to_csv(OUTPUT_FOLDER / 'marginal_cost.csv', index=False)
#        print(f'{marginal_cost_df.head()=}')
#
#        # Test production dataframe for a participant
#        participant_key = 'wind'
#        production_df = self.mock_comparative_statics.get_dataframe('production', participant_key)
#        assert_dataframe(production_df)
#        # Save the dataframe to a files
#        production_df.to_csv(OUTPUT_FOLDER / 'production_wind.csv', index=False)
#
#        # Test variable costs dataframe for a participant
#        variable_costs_df = self.mock_comparative_statics.get_dataframe('variable_costs', 'thermal')
#        assert_dataframe(variable_costs_df)
#        # Save the dataframe to a files
#        variable_costs_df.to_csv(OUTPUT_FOLDER / 'variable_costs_thermal.csv', index=False)


if __name__ == '__main__':
    unittest.main()

