from pathlib import Path

from src.utils.logging_config import setup_logging
setup_logging(level = "INFO")
from src.finalization_module import SimulationFinalizer, visualize
from src.constants import BASE_PATH


simulation_name: str = "zero_hydro"

simulation_folder: Path = BASE_PATH / "sim" / simulation_name
x_variable: dict[str, str] = {"name": "hydro_factor", "label": "Hydro factor"}
participants: list[str] = ["solar", "wind", "thermal"]

simulation = SimulationFinalizer(simulation_folder, x_variable, participants)

visualize(simulation)

