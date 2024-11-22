#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=2:30:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=5G 
#SBATCH --job-name=process_expensive_blackout
#SBATCH --output=/projects/p32342/slurm/process_expensive_blackout.out
#SBATCH --error=/projects/p32342/slurm/process_expensive_blackout.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load python-miniconda3/4.12.0
cd /projects/p32342/code
python expensive_blackout.py
