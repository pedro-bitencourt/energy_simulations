#!/bin/bash
#SBATCH --account=p32342
#SBATCH --partition=normal
#SBATCH --time=10:00:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=40G 
#SBATCH --job-name=process_job
#SBATCH --output=/projects/p32342/process.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=pdm6134

python main.py process


