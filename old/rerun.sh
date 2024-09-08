#!/bin/bash
#SBATCH --account=p32342
#SBATCH --partition=short 
#SBATCH --time=1:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=2G 
#SBATCH --job-name=rerun_job
#SBATCH --output=/projects/p32342/rerun.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu

python main.py rerun


