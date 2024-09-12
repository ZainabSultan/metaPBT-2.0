#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=06:00:00             # Wall time limit (hh:mm:ss)

#SBATCH --mem-per-cpu=2GB           # Memory per CPU core
##SBATCH --mem-per-cpu=4GB           # Memory per CPU core
##SBATCH --mem-per-cpu=5GB

#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

source ${HOME}/.bashrc
conda activate dkl_env

cluster_logs_dir=$(ws_find dkl_exps)/cluster_logs/${SLURM_JOB_NAME}/
echo $cluster_logs_dir 


cd /home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/

kwargs="--ws_dir=$cluster_logs_dir --seed=0 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}' --meta_job_logs_dirs=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c5/2024-09-03_PPO_gravity_0.00125_pb2_dkl_Size8_CARLMountainCar_timesteps_total /pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c10/2024-09-03_PPO_gravity_0.0025_pb2_dkl_Size8_CARLMountainCar_timesteps_total"
echo $kwargs
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=1 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}' --meta_job_logs_dirs=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c5/2024-09-03_PPO_gravity_0.00125_pb2_dkl_Size8_CARLMountainCar_timesteps_total /pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c10/2024-09-03_PPO_gravity_0.0025_pb2_dkl_Size8_CARLMountainCar_timesteps_total"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=2 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}' --meta_job_logs_dirs=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c5/2024-09-03_PPO_gravity_0.00125_pb2_dkl_Size8_CARLMountainCar_timesteps_total /pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c10/2024-09-03_PPO_gravity_0.0025_pb2_dkl_Size8_CARLMountainCar_timesteps_total"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=3 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=4 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=5 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=6 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=7 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=8 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs

kwargs="--ws_dir=$cluster_logs_dir --seed=9 --num_samples=8 --method=pb2_meta_dkl --context='{\"gravity\":0.00025}'"
python3 -m ppo_metaDKL $kwargs




