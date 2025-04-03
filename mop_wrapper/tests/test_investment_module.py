import unittest
from pathlib import Path
from src.investment_module import InvestmentProblem


class TestInvestmentProblem(unittest.TestCase):
    def setUp(self):
        self.test_folder = '/Users/pedrobitencourt/quest/renewables/data/salto_capacity/'
        self.xml_basefile = '/Users/pedrobitencourt/quest/renewables/data/salto_capacity/inputs/2024.xml'
        
        # Define test parameters
        self.exogenous_variables = {
            'hydro_factor': {'pattern': 'HYDRO_FACTOR', 'value': 0.75},
        }
        self.endogenous_variables = {
            'wind': {'pattern': 'WIND_CAPACITY', 'initial_guess': 1000},
            'solar': {'pattern': 'SOLAR_CAPACITY', 'initial_guess': 1000},
            'thermal': {'pattern': 'THERMAL_CAPACITY', 'initial_guess': 300}
        }
        self.general_parameters = {
            'daily': True,
            'name_subfolder': 'CAD-2024-DIARIA',
            'xml_basefile': self.xml_basefile,
            'annual_interest_rate': 0.0,
            'years_run': 6.61,
            'requested_time_run': 3.5
        }


    def test_initial_profits_computation(self):
        """Test that profits can be computed at the initial guess"""
        # Initialize the problem
        problem = InvestmentProblem(
            parent_folder=self.test_folder,
            exogenous_variables=self.exogenous_variables,
            endogenous_variables=self.endogenous_variables,
            general_parameters=self.general_parameters
        )

        # Get the initial investment from the first iteration
        initial_iteration = problem.optimization_trajectory[0]
        initial_investment = initial_iteration.capacities

        # Compute profits and derivatives
        profits, derivatives = problem.profits_and_derivatives(initial_investment)

        # Verify the structure of returned values
        self.assertIsInstance(profits, dict)
        self.assertIsInstance(derivatives, dict)

        # Check that we have profits and derivatives for each endogenous variable
        for var in self.endogenous_variables.keys():
            self.assertIn(var, profits)
            self.assertIn(var, derivatives)
            self.assertIsInstance(profits[var], (int, float))
            self.assertIsInstance(derivatives[var], (int, float))

        # Verify that profits are finite numbers
        for profit in profits.values():
            self.assertFalse(profit is None)
            self.assertFalse(profit is float('inf'))
            self.assertFalse(profit is float('-inf'))
            self.assertFalse(profit != profit)  # Check for NaN


if __name__ == '__main__':
    unittest.main()
