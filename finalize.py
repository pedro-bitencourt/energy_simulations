from pathlib import Path

from src.utils.logging_config import setup_logging

setup_logging(level = "info")

from src.finalization_module import SimulationFinalizer, visualize
from src.constants import BASE_PATH

simulation_name: str = "factor_compartir_gas_hi_wind"
costs_path: Path = BASE_PATH / "code/cost_data/gas_high_wind_cost.json"


simulation_folder: Path = BASE_PATH / "sim" / simulation_name
x_variable: dict[str, str] = {"name": "factor_compartir", "label": "Factor Compartir"}
endogenous_participants: list[str] = ["solar", "wind", "thermal"]

simulation = SimulationFinalizer(simulation_folder,
                                 x_variable,
                                 endogenous_participants,
                                 costs_path,
                                 overwrite = False)
visualize(simulation)

