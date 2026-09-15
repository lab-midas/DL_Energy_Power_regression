#!/bin/bash

####

# Define slurm job parameters

####

#SBATCH --job-name=Transformer_dataset
#SBATCH --cpus-per-task=8          # Number of CPU cores to allocate
#SBATCH --mem=128G                 # RAM allocation
#SBATCH --partition=day
#SBATCH --gpus=1
#SBATCH --time=08:00:00            # Maximum hours
#SBATCH --nodelist=node-gpu-02
#SBATCH --output=logs/Transformer_dataset.%J.out
#SBATCH --error=logs/Transformer_dataset.%J.err

# Initialize Conda for the shell
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the mri_data Conda environment
conda activate mri_data

# Print the GPU node hostname
echo "Running on node: $(hostname)"

# Run the Python script
python dataset.py