'''
Unit tests for the run_module.py and run_processor_module.py
'''
import unittest
import numpy as np
from pathlib import Path
from investment_module import InvestmentProblem

class TestInvestmentProblem(unittest.TestCase):
    def setUp(self):
        mock_investment_problem_name: str = 'mock_investment_problem'
        mock_experiment_name: str = 'mock_experiment'
        mock_exogenous_variables: dict = {'hydro': {'pattern': 'HYDRO_FACTOR'}} 
        mock_endogenous_variables: dict = {'wind_capacity': {'value': 0.5, 'pattern': 'WIND_CAPACITY'}}
        mock_exogenous_variables_grid: list[dict] = [{'hydro': 0.1}, {'hydro': 0.2}] 
        xml_basefile: str = 'code/xml/inv_zero_mc_thermal.xml'
        mock_general_parameters: dict = {'daily': False,
                                'name_subfolder': 'PRUEBA',
                                'xml_basefile': xml_basefile}
        self.mock_investment_problem = InvestmentProblem(mock_investment_problem_name,
                                                   mock_experiment_name,
                                                   mock_exogenous_variables,
                                                   mock_endogenous_variables,
                                                   mock_exogenous_variables_grid,
                                                   mock_general_parameters)

    def tearDown(self):
        pass

    def test_run_on_quest(self):  
        job_id = self.mock_investment_problem.run_on_quest()
        # assert job_id is not None
        self.assertTrue(job_id is not None)

         
        






if __name__ == '__main__':
    unittest.main()
