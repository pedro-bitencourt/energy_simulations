'''
Implements unit testing for the experiment_module.py module.

UNFINISHED
'''

import unittest
import os
import time
import numpy as np

from experiment_module import Experiment, ExperimentVisualizer
from constants import BASE_PATH


#class TestExperimentModule(unittest.TestCase):
#    def setUp(self):
#        name: str = "mrs_experiment"
#
#        # general parameters
#        general_parameters: dict = {
#                "xml_basefile": f"{MOCK_BASE_PATH}/code/xml/mrs_experiment.xml"
#        }
#
#        # to change
#        grid_hydro: list = [0.1, 0.2, 0.3]
#        grid_thermal: list = [0.1, 0.2, 0.3]
#        variables_grid: dict[str, np.ndarray] = {
#                'hydro_factor': grid_hydro, 
#                'thermal_capacity': grid_thermal
#        }
#
#        # initialize the experiment
#        self.mock_experiment = Experiment(name, variables_grid, general_parameters)
#
#
#    def test_submit_jobs(self):
#        job_ids: list = self.mock_experiment.submit_experiment()
#
#        # sleep for 5 minutes
#        time.sleep(300)
#
#        # check if the jobs are still running
#        for job_id in job_ids:
#            job_ok: bool = is_job_running(job_id)
#            self.assertTrue(job_ok)
#        

# test the ExperimentVisualizer class 
class TestExperimentVisualizer(unittest.TestCase):
    def setUp(self):
        name: str = "mrs_experiment"

        # general parameters
        run_name_function_params = {'hydro_factor': {'position': 0, 'multiplier': 10},
                                    'thermal_capacity': {'position': 1, 'multiplier': 1}}
        general_parameters: dict = {
            "xml_basefile": f"{BASE_PATH}/code/xml/mrs_experiment.xml",
            "name_function": run_name_function_params
        }
        # to change
        grid_hydro: np.ndarray = np.linspace(0.2, 1, 5)
        grid_thermal: np.ndarray = np.linspace(8, 100, 5)

        variables_grid: dict[str, np.ndarray] = {
                'hydro_factor': grid_hydro, 
                'thermal_capacity': grid_thermal
        }

        variables: dict = {
                        "hydro_factor": {"pattern": "HYDRO_FACTOR*"},
                        "thermal_capacity": {"pattern": "THERMAL_CAPACITY"}
                    }


        # initialize the experiment
        mock_experiment = Experiment(name, variables, variables_grid, general_parameters)

        # initialize the visualizer object
        self.mock_visualizer = ExperimentVisualizer(mock_experiment)

    # test the plot_heatmap method
    def test_plot_heatmaps(self):
        self.mock_visualizer.plot_heatmaps()
        # check if the plot was saved
        self.assertTrue(
            os.path.exists(
                f'{MOCK_BASE_PATH}/result/{self.mock_visualizer.experiment.name}_heatmap.png'
            )
        )

    # test the plot_stacked_price_distributions method
    def test_plot_intraweek_price_distributions(self):
        self.mock_visualizer.plot_intraweek_price_distributions()
        # check if the plot was saved
        name: str = self.mock_visualizer.experiment.name
        self.assertTrue(os.path.exists(f'{MOCK_BASE_PATH}/result/{name}_price_distributions.png'))
    


if __name__ == '__main__':
    unittest.main()
