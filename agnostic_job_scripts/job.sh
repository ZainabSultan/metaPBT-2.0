#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
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

# Loop over seeds and run jobs with the same context
context='{"TERRAIN_STEP":0.13999999999999999}'

kwargs="--ws_dir=$tmp_logs_dir --seed=5 --env_name=CARLBipedalWalker --num_samples=8 --scheduler=pb2 --context='$context' --max=1000000 --t_ready=50000"
echo "Running job with seed=$seed and context=$context"
python3 -m DKL_experiment $kwargs


 #!/bin/bash
# #SBATCH --partition=single
# #SBATCH --nodes=1                   # Number of nodes
# #SBATCH --ntasks=8                  # Number of tasks (or processes) per node
# #SBATCH --cpus-per-task=8          # Number of CPU cores per task
# ##SBATCH --gres=gpu:1                # Number of GPUS
# #SBATCH --time=12:00:00             # Wall time limit (hh:mm:ss)

# #SBATCH --mem-per-cpu=2GB           # Memory per CPU core
# ##SBATCH --mem-per-cpu=4GB           # Memory per CPU core
# ##SBATCH --mem-per-cpu=5GB

# #SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
# #SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

# source ${HOME}/.bashrc
# conda activate dkl_env


# # copy the metadata from the workspace to the tmpdir
# ##cp -r $(ws_find dkl_exps)/metadata ${TMPDIR}/metadata

# #set up all the args
# ##kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cpu" --working-dir=$(ws_find dkl_exps) --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 
# ##--warmstart-smbo"
# ##kwargs="--experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" --working-dir=$(ws_find automl) \
# ##--datasetpath=${TMPDIR}/data --metadata-file=${TMPDIR}/metadata/deepweedsx_balanced-epochs-trimmed.csv" 

# ##python3 -m ppo_ray_example $kwargs # --experiment-name=$SLURM_JOB_NAME --n-workers=$SLURM_CPUS_PER_TASK --device="cuda" # --warmstart-smbo



# kwargs="--ws_dir=$cluster_logs_dir --seed=0 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs
# kwargs="--ws_dir=$cluster_logs_dir --seed=1 --num_samples=8 --method="pb2" --method="pb2"  --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs

# kwargs="--ws_dir=$cluster_logs_dir --seed=2 --num_samples=8 --method="pb2"  --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs
# kwargs="--ws_dir=$cluster_logs_dir --seed=3 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs

# kwargs="--ws_dir=$cluster_logs_dir --seed=4 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs
# kwargs="--ws_dir=$cluster_logs_dir --seed=5 --num_samples=8 --method="pb2"  --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs


# kwargs="--ws_dir=$cluster_logs_dir --seed=6 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs
# kwargs="--ws_dir=$cluster_logs_dir --seed=7 --num_samples=8 --method="pb2"  --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs


# kwargs="--ws_dir=$cluster_logs_dir --seed=8 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs
# kwargs="--ws_dir=$cluster_logs_dir --seed=9 --num_samples=8 --method="pb2" --context='{"gravity": 0.00325}'"
# python3 -m ppo_ray_example $kwargs

