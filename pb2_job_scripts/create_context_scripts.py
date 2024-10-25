import os
import numpy as np
# USAGE change env name, featurename 
# Directory where the files will be created

## chnage these

feature='GRAVITY_Y'
env_name='CARLLunarLander'

difficult_envs = ['CARLBipedalWalker', 'CARLLunarLander']

directory = f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/pb2_job_scripts/{env_name}/{feature}/'
defaults_cartpole={'length':0.5, 'tau':0.02, 'gravity':9.8}
defaults_mcd={'gravity': 0.0025}
defaults_lunar_lander = {
    "FPS": 50,
    "SCALE": 30.0,
    "MAIN_ENGINE_POWER": 13.0,
    "SIDE_ENGINE_POWER": 0.6,
    "INITIAL_RANDOM": 1000.0,
    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,
    "LEG_AWAY": 20,
    "LEG_DOWN": 18,
    "LEG_W": 2,
    "LEG_H": 8,
    "LEG_SPRING_TORQUE": 40,
    "SIDE_ENGINE_HEIGHT": 14.0,
    "SIDE_ENGINE_AWAY": 12.0,
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400
}
defaults_bipedal = {
    "FPS": 50,
    "SCALE": 30.0,
    "GRAVITY_X": 0,
    "GRAVITY_Y": -10,
    "FRICTION": 2.5,
    "TERRAIN_STEP": 14 / 30.0,
    "TERRAIN_LENGTH": 200,
    "TERRAIN_HEIGHT": 600 / 30 / 4,
    "TERRAIN_GRASS": 10,
    "TERRAIN_STARTPAD": 20,
    "MOTORS_TORQUE": 80,
    "SPEED_HIP": 4,
    "SPEED_KNEE": 6,
    "LIDAR_RANGE": 160 / 30.0,
    "LEG_DOWN": -8 / 30.0,
    "LEG_W": 8 / 30.0,
    "LEG_H": 34 / 30.0,
    "INITIAL_RANDOM": 5,
    "VIEWPORT_W": 600,
    "VIEWPORT_H": 400
}

if env_name == 'CARLMountainCar':
    DEFAULT_VALUE = defaults_mcd[feature]
elif env_name=='CARLCartPole':
    DEFAULT_VALUE = defaults_cartpole[feature]
elif env_name=='CARLBipedalWalker':
    DEFAULT_VALUE = defaults_bipedal[feature]
elif env_name=='CARLLunarLander':
    DEFAULT_VALUE = defaults_lunar_lander[feature]

if env_name in difficult_envs:
    max=1000_000
    t_ready = 50_000
    time='30:00:00'
else:
    max=100_000
    t_ready = 50_00
    time='3:00:00'
#cartpo9le
# tau 0.02
# pole lenght 0.5
# Length values based on script index
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
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
##SBATCH --gres=gpu:1                # Number of GPUS
#SBATCH --time={time}            # Wall time limit (hh:mm:ss)
#SBATCH --mem-per-cpu=10GB           # Memory per CPU core

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
    kwargs="--ws_dir=$tmp_logs_dir --seed=$seed --env_name={env_name} --num_samples=4 --scheduler=pb2 --context='$context' --max={max} --t_ready={t_ready}"
    echo "Running job with seed=$seed and context=$context"
    python3 -m DKL_experiment $kwargs
    rsync -av $tmp_logs_dir $cluster_logs_dir
done
"""

    with open(filepath, 'w') as file:
        file.write(script_content)
    
    print(f"Created: {filepath}")
