#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=60:00:00            # Wall time limit (hh:mm:ss)

#SBATCH --mem-per-cpu=4GB           # Memory per CPU core
##SBATCH --mem-per-cpu=5GB

#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

# Activate environment
source ${HOME}/.bashrc
conda activate dkl_env

# Set the working directory for the logs
cluster_logs_dir=$(ws_find dkl_exps)/cluster_logs/${SLURM_JOB_NAME}/
tmp_logs_dir=$TMPDIR/${SLURM_JOB_NAME}/
on_exit() {
    rsync -av $tmp_logs_dir $cluster_logs_dir
}
trap on_exit EXIT

# Change to project directory
cd /home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/


# Run the job for context 1
context='{"LEG_H":0.8}'
kwargs="--env_name=CARLLunarLander --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 2
context='{"LEG_H":14.4}'
kwargs="--env_name=CARLLunarLander --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 3
context='{"LEG_H":10.4}'
kwargs="--env_name=CARLLunarLander --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 4
context='{"LEG_H":5.6}'
kwargs="--env_name=CARLLunarLander --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 5
context='{"LEG_H":17.6}'
kwargs="--env_name=CARLLunarLander --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs
