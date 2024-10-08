#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=8                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)

#SBATCH --mem-per-cpu=2GB           # Memory per CPU core
##SBATCH --mem-per-cpu=4GB           # Memory per CPU core
##SBATCH --mem-per-cpu=5GB

#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

source ${HOME}/.bashrc
conda activate dkl_env


# copy the metadata from the workspace to the tmpdir
##cp -r $(ws_find dkl_exps)/metadata ${TMPDIR}/metadata

#set up all the args
##kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cpu" --working-dir=$(ws_find dkl_exps) --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 
##--warmstart-smbo"
##kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" --working-dir=$(ws_find automl) \
##--datasetpath=${TMPDIR}/data --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 

##python3 -m ppo_ray_example $kwargs # --experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" # --warmstart-smbo

# reserve whole node

kwargs="--ws_dir=$cluster_logs_dir --seed=0 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs
kwargs="--ws_dir=$cluster_logs_dir --seed=1 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=2 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs
kwargs="--ws_dir=$cluster_logs_dir --seed=3 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=4 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs
kwargs="--ws_dir=$cluster_logs_dir --seed=5 --num_samples=8  --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs


kwargs="--ws_dir=$cluster_logs_dir --seed=6 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs
kwargs="--ws_dir=$cluster_logs_dir --seed=7 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs


kwargs="--ws_dir=$cluster_logs_dir --seed=8 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs
kwargs="--ws_dir=$cluster_logs_dir --seed=9 --num_samples=8 --context='{"gravity": 0.00325}'"
python3 -m ppo_ray_example $kwargs

