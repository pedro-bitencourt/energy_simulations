import sys
from pathlib import Path
import pandas as pd
sys.path.append('/projects/p32342/code')

from src.solver_module import Solver
from src.data_analysis_module import conditional_means

name: str = 'zero_hydro'
xml_basefile: str = '/projects/p32342/code/xml/zero_hydro.xml'
costs_path: str = '/projects/p32342/data/cost_original.json'


general_parameters: dict = {
    'daily': True,
    'xml_basefile': xml_basefile,
    'cost_path': costs_path,
    'annual_interest_rate': 0.0,
    'email': 'pedro.bitencourt@u.northwestern.edu'
}

exogenous_variable: dict[str, dict] = {
    'hydro_present': {'value': 0},
}
endogenous_variables: dict[str, dict] = {
    'wind_capacity': {'initial_guess': 2000},
    'solar_capacity': {'initial_guess': 1500},
    'thermal_capacity': {'initial_guess': 1200}
}

parent_folder: Path = Path('/projects/p32342/sim/zero_hydro')
parent_folder.mkdir(parents=True, exist_ok=True)

solver: Solver = Solver(parent_folder, exogenous_variable,
                        endogenous_variables, general_parameters)

####
solver_results = solver.solver_results()
print(solver_results)
solver_results_df = pd.DataFrame([solver_results])
solver_results_df.to_csv("/projects/p32342/sim/zero_hydro/solver_results.csv")

last_run = solver.last_run()

run_df = last_run.full_run_df()
solver.paths['raw'].parent.mkdir()
run_df.to_csv(solver.paths['raw'])
