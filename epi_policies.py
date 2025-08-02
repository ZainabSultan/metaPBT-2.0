import os
import numpy as np
# USAGE chzahnhge env name, featurename 
# Directory where the files will be created

## chnage these
feature='length'
env_name='CARLCartPole'



directory = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/pb2_job_scripts/holder/{feature}'.format(feature=feature)
defaults_cartpole={'length':0.5, 'tau':0.02, 'gravity':9.8}
defaults_mcd={'gravity': 0.0025}
if env_name == 'CARLMountainCar':
    DEFAULT_VALUE = defaults_mcd[feature]
elif env_name=='CARLCartPole':
    DEFAULT_VALUE = defaults_cartpole[feature]
feature_values = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
    2.1, 2.2
]) * DEFAULT_VALUE
home_str = r'${HOME}'
# Generate scripts
for i, value in enumerate(feature_values, start=1):
    filename = f"context_{i}.sh"
    filepath = os.path.join(directory, filename)
    
    # Create the script content
    script_content = f"""#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (or processes) per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time=30:00:00             # Wall time limit (hh:mm:ss)

##SBATCH --mem-per-cpu=2GB           # Memory per CPU core
#SBATCH --mem-per-cpu=4GB           # Memory per CPU core
##SBATCH --mem-per-cpu=5GB

#SBATCH --error=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.err
#SBATCH --output=/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/%x/%j.out

# Activate environment
source {home_str}/.bashrc
conda activate dkl_env

# Set the working directory for the logs
cluster_logs_dir=$(ws_find dkl_exps)/cluster_logs/${{SLURM_JOB_NAME}}/
tmp_logs_dir=$TMPDIR/${{SLURM_JOB_NAME}}/
on_exit() {{
    rsync -av $tmp_logs_dir $cluster_logs_dir

}}
trap on_exit EXIT
# Change to project directory
cd /home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/

# Loop over seeds and run jobs with the same context
context='{{"{feature}":{value}}}'
for seed in {{0..9}}; do
    kwargs="--ws_dir=$tmp_logs_dir --seed=$seed --env_name={env_name} --num_samples=8 --scheduler=pb2 --context='$context'"
    echo "Running job with seed=$seed and context=$context"
    python3 -m metaDKL_experiment $kwargs
done
"""
    with open(filepath, 'w') as file:
        file.write(script_content)
    
    print(f"Created: {filepath}")
