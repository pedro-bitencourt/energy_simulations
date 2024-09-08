#!/bin/bash
#SBATCH --account=p32342
#SBATCH --partition=short
#SBATCH --time=2:30:00 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=60G 
#SBATCH --job-name=/projects/p32342/temp/static_20y_full/3200_150_2250.xml
#SBATCH --output=/projects/p32342/output/static_20y_full/3200_150_2250.out

module purge
module load wine/6.0.1
cd /projects/p32342/software/Ver_2.3
wine "Z:\projects\p32342\software\Java\jdk-11.0.22+7\bin\java.exe" -Xmx36G -jar MOP_Mingo.JAR "Z:\projects\p32342\code\xml\test.xml"

