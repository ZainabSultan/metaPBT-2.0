#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=30:00:00             # Wall time limit (hh:mm:ss)

#SBATCH --mem-per-cpu=2GB           # Memory per CPU core
##SBATCH --mem-per-cpu=4GB           # Memory per CPU core
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

# Loop over seeds and run jobs with the same context
context='{"tau":0.024}'
for seed in {0..9}; do
    kwargs="--ws_dir=$tmp_logs_dir --seed=$seed --env_name=CARLCartPole --num_samples=8 --scheduler=pb2 --context='$context'"
    echo "Running job with seed=$seed and context=$context"
    python3 -m DKL_experiment $kwargs
done
