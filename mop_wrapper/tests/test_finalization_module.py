import unittest
from ..src.finalization_module_new2 import SimulationData, build_simulation_data

from ..src.utils.logging_config import setup_logging
from ..src.constants import BASE_PATH

from pathlib import Path

setup_logging()

class TestFinalization(unittest.TestCase): 
    def setUp(self):
        sim_name: str = 'factor_compartir_gas_hi_wind'
        participants: Tuple[str] = ('wind', 'solar', 'thermal')
        x_variable: Dict[str, Any] = {
            'name': 'factor_compartir',
            'label': 'Factor Compartir'
        }
        cost_parameters_file: Path = BASE_PATH / 'code' / 'cost_data' / 'gas_high_wind.json'
        self.sim_data = build_simulation_data(sim_name, participants, x_variable, cost_parameters_file)

    def test_build_simulation_data(self):
        print(self.sim_data)
