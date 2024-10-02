#!/bin/bash
#SBATCH --account=b1048
#SBATCH --partition=b1048
#SBATCH --time=4:30:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=10G 
#SBATCH --job-name=inv_zero_mc_thermal_test
#SBATCH --output=/projects/p32342/output/test.out
#SBATCH --error=/projects/p32342/output/test.err
#SBATCH --mail-user=pedro.bitencourt@u.northwestern.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=qhimem[0207-0208]

module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
wine "Z:\projects\p32342\software\Java\jdk-11.0.22+7\bin\java.exe" -Xmx10G -jar MOP_Mingo.JAR "Z:\projects\p32342\code\xml\test\zero_mc_thermal.xml"
