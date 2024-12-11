import unittest
from pathlib import Path
import numpy as np
import pandas as pd



from ..comparative_statics_module import ComparativeStatics
from ..constants import DATETIME_FORMAT, SCENARIOS

from ..logging_config import setup_logging
setup_logging()




OUTPUT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data/results/salto_capacity_v3')
PARENT_FOLDER: Path = Path('/Users/pedrobitencourt/energy_simulations/tests/data')

REQUESTED_TIME_SOLVER = 1
REQUESTED_TIME_RUN = 1

COLUMNS_TO_CHECK = ['run','datetime'] + SCENARIOS



class TestComparativeStatics(unittest.TestCase):
    def setUp(self):
        name: str = 'salto_capacity_v3'
        xml_basefile: str = '/home/pdm6134/salto_capacity_v2.xml'
        
        general_parameters: dict[str, Union[bool, str, float, int]] = {
            'daily': True,
            'name_subfolder': 'CAD-2024-DIARIA',
            'xml_basefile': xml_basefile,
            'email': 'pedro.bitencourt@u.northwestern.edu',
            'annual_interest_rate': 0.0,
            'requested_time_run': 7.5,
            'requested_time_solver': 18.5
        }
        
        exogenous_variable_name: str = 'salto_capacity'
        exogenous_variable_pattern: str = 'SALTO_CAPACITY'
        exogenous_variable_label: str = 'Salto Capacity'
        exogenous_variable_grid: list[float] = [0, 101.25]
        exogenous_variables: dict[str, dict] = {
            exogenous_variable_name: {
                'pattern': exogenous_variable_pattern,
                'label': exogenous_variable_label,
                'grid': exogenous_variable_grid
            }
        }
        
        endogenous_variables: dict[str, dict[str, Union[str, list[int]]]] = {
            'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': [7000, 5112]},
            'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': [2489, 1907]},
            'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': [1309, 1123]}
        }
        
        variables: dict[str, dict] = {
            'exogenous': exogenous_variables,
            'endogenous': endogenous_variables
        }
        # Create the ComparativeStatics object
        self.mock_comparative_statics = ComparativeStatics(name,
                                                 variables,
                                                 general_parameters,
                                                    PARENT_FOLDER)
        
    def tearDown(self):
        # delete all .csv files in output folder
        #for file in OUTPUT_FOLDER.glob('*.csv'):
        #    file.unlink(missing_ok=True)
        pass

    def test_process(self):
        self.mock_comparative_statics.process()
        # Check if the output folder contains the expected files
        expected_files = ['random_variables.csv', 'conditional_means.csv', 'investment_results.csv']
        for file in expected_files:
            self.assertTrue((OUTPUT_FOLDER / file).exists(),
                            f"Expected file '{file}' not found in the output folder '{OUTPUT_FOLDER}'")
        print(f'{list(OUTPUT_FOLDER.glob("*.csv"))=}')


if __name__ == '__main__':
    unittest.main()

