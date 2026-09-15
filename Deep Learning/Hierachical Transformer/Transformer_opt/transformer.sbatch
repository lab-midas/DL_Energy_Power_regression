#!/bin/bash

####
# Define slurm job parameters
####

#SBATCH --job-name=Transformer
#SBATCH --cpus-per-task=16           # Number of CPU cores to allocate
#SBATCH --mem=256G                   # RAM allocation
#SBATCH --partition=month
#SBATCH --gpus=1
#SBATCH --time=31-00:00:00
#SBATCH --nodelist=node-gpu-05
#SBATCH --output=logs/Transformer.%j.out
#SBATCH --error=logs/Transformer.%j.err

# Initialize Conda for the shell
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the mri_data Conda environment
conda activate mri_data

# Print the GPU node hostname
echo "Running on node: $(hostname)"

# Start the Dash app in the background
python transformer.py