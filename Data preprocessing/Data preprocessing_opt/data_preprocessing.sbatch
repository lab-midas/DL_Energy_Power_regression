#!/bin/bash

####
# Define slurm job parameters
####

#SBATCH --job-name=data_preprocessing
#SBATCH --cpus-per-task=12
#SBATCH --partition=day
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --nodelist=node-gpu-06
#SBATCH --error=logs/job.%J.%x.err
#SBATCH --output=logs/job.%J.%x.out

# Initialize Conda for the shell
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the mri_data Conda environment
conda activate mri_data

# Print the GPU node hostname
echo "Running on node: $(hostname)"

# Start the Dash app in the background
python data_preprocessing.py