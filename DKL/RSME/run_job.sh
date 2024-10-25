#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=2           # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=10:00:00               # Wall time limit (hh:mm:ss)
#SBATCH --mem-per-cpu=32GB           # Memory per CPU core
#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

# Activate environment
source ${HOME}/.bashrc
conda activate dkl_env

# Set the working directory for the logs
# 

# Change to project directory
cd /home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/

# Run the experiment with the context and seed
python3 -m experiments_DKL_hps
