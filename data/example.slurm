#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --output=ScriptOutput.log

# Make sure to set up parameters relevant to your cluster above
# Activate your conda environment
source activate myenv

# Run the pipeline
srun python data_pipeline/base_script.py
