#!/bin/bash
#SBATCH --job-name=myAnalysisName
#SBATCH --mail-type=ALL                       
#SBATCH --mail-user=timothygao@berkeley.edu
#SBATCH -o myAnalysisName.out #File to which standard out will be written
#SBATCH -e myAnalysisName.err #File to which standard err will be written

eval "$(conda shell.bash hook)"
conda activate llm

python3 peturb_stats.py
