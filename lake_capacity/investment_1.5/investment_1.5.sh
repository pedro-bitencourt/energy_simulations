#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --job-name=investment_1.5
#SBATCH --output=lake_capacity/investment_1.5.out
#SBATCH --error=lake_capacity/investment_1.5.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0


python - <<END

print('Solving the investment_problem problem investment_1.5')

import sys
import json
sys.path.append('/projects/p32342/code')
from src.investment_module import InvestmentProblem

with open('lake_capacity/investment_1.5/investment_1.5_data.json', 'r') as f:
    investment_data = json.load(f)
investment_problem = InvestmentProblem(**investment_data)
print('Successfully loaded the investments problem data')
sys.stdout.flush()
sys.stderr.flush()
investment_problem.solve_investment()
END
