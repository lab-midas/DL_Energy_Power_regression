#!/bin/bash

#####
# SLURM JOB PARAMETERS
#####

#SBATCH --job-name=Transformer_dataset
#SBATCH --cpus-per-task=8           # Number of CPU cores to allocate
#SBATCH --mem=128G                  # RAM allocation
#SBATCH --partition=day
#SBATCH --gpus=1
#SBATCH --time=08:00:00            # Maximum hours
#SBATCH --nodelist=node-gpu-02
#SBATCH --output=logs/Transformer_dataset.%J.out
#SBATCH --error=logs/Transformer_dataset.%J.err

# Initialize Conda for the shell
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate mri_data

# Print the GPU node hostname
echo "Running on node: $(hostname)"

# Specify the path to the notebook you want to execute
NOTEBOOK_PATH="/home/raraabf1/PhD/MRI_data_analysis/Deep Learning/Pipeline-V2/Hierachical Transformer/Dataset Tokenization/Dataset_opt/dataset.ipynb"

# Run Jupyter notebook and execute the specified notebook
jupyter nbconvert --to notebook --execute "$NOTEBOOK_PATH" --output "$NOTEBOOK_PATH"