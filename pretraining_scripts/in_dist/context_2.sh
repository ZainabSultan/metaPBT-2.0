#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=30:00:00             # Wall time limit (hh:mm:ss)

##SBATCH --mem-per-cpu=2GB           # Memory per CPU core
#SBATCH --mem-per-cpu=4GB           # Memory per CPU core
##SBATCH --mem-per-cpu=5GB

#SBATCH --error=$TMPDIR/${SLURM_JOB_NAME}/cluster_logs/%x/%j.err
#SBATCH --output=$TMPDIR/${SLURM_JOB_NAME}/cluster_logs/%x/%j.out

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
dir_list=(
    "/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/pb2.gravity.c6/2024-09-15_18:58:50_PPO_gravity_5.88_pb2_Size8_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_gravity_5.88"
)
# Loop over seeds and run jobs with the same context
context='{"gravity":5.88}'
for seed in {0..9}; do
    kwargs="--ws_dir=$tmp_logs_dir --seed=$seed --env_name=CARLCartPole --num_samples=8 --metadata_dir_list "${dir_list[@]}" --context='$context'"
    echo "Running job with seed=$seed and context=$context"
    python3 -m metaDKL_experiment $kwargs
done
