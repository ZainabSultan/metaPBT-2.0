#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=30:00:00            # Wall time limit (hh:mm:ss)

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




# Run the job for context 2
context='{"TERRAIN_STEP":0.09333333333333334}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 1
context='{"TERRAIN_STEP":0.04666666666666667}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 3
context='{"TERRAIN_STEP":0.13999999999999999}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 4
context='{"TERRAIN_STEP":0.18666666666666668}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 5
context='{"TERRAIN_STEP":0.23333333333333334}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 6
context='{"TERRAIN_STEP":0.27999999999999997}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 7
context='{"TERRAIN_STEP":0.32666666666666666}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 8
context='{"TERRAIN_STEP":0.37333333333333335}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 9
context='{"TERRAIN_STEP":0.42000000000000004}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 10
context='{"TERRAIN_STEP":0.4666666666666667}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 11
context='{"TERRAIN_STEP":0.5133333333333334}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 12
context='{"TERRAIN_STEP":0.5599999999999999}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 13
context='{"TERRAIN_STEP":0.6066666666666667}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 14
context='{"TERRAIN_STEP":0.6533333333333333}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 15
context='{"TERRAIN_STEP":0.7}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 16
context='{"TERRAIN_STEP":0.7466666666666667}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 17
context='{"TERRAIN_STEP":0.7933333333333333}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 18
context='{"TERRAIN_STEP":0.8400000000000001}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 19
context='{"TERRAIN_STEP":0.8866666666666666}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 20
context='{"TERRAIN_STEP":0.9333333333333333}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 21
context='{"TERRAIN_STEP":0.9800000000000001}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs

# Run the job for context 22
context='{"TERRAIN_STEP":1.0266666666666668}'
kwargs="--env_name=CARLBipedalWalker --context='$context'"
echo "Running job with context=$context"
python3 -m DKL.EPI.generate_dataset $kwargs
