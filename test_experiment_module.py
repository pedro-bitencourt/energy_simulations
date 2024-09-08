'''
Implements unit testing for the experiment_module.py module.
'''

import unittest
from experiment_module import Experiment

MOCK_BASE_PATH = '/quest/renewables'

class TestExperimentModule(unittest.TestCase):
    def setUp(self):
        name: str = "mrs_experiment"

        # general parameters
        general_parameters: dict = {
                "xml_basefile": f"{MOCK_BASE_PATH}/code/xml/mrs_experiment.xml"
        }

        # to change
        grid_hydro: list = [0.1, 0.2, 0.3]
        grid_thermal: list = [0.1, 0.2, 0.3]

        # construct the variables grid
        variables_grid = []
        for hydro in grid_hydro:
            for thermal in grid_thermal:
                variables_temp: dict = {
                        "hydro_factor": {"pattern": "HYDRO_FACTOR*", "value": hydro},
                        "thermal": {"pattern": "THERMAL", "value": thermal}
                }
                variables_grid.append(variables_temp)

        # initialize the experiment
        self.mock_experiment = Experiment(name, variables_grid, general_parameters)


    def test_submit_jobs(self):
        successful: bool = self.mock_experiment.submit_experiment()
        self.assertTrue(successful)
