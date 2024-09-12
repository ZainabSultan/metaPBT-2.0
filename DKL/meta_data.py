import os
import csv
import re
import argparse
from typing import List, Dict, Any
import json
import pandas as pd
from ast import literal_eval
import numpy as np

# Function to extract context from directory name
def extract_context_from_dir_name(dir_name: str) -> List[float]:
    parts = dir_name.split('_')
    
    seed_index = None
    for i, part in enumerate(parts):
        if part.startswith('seed'):
            seed_index = i
            break
    
    if seed_index is not None and seed_index < len(parts) - 1:
        context_parts = parts[seed_index + 1:]
        context_values = []

        for part in context_parts:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", part)
            context_values.extend([float(num) for num in numbers])

        return context_values
    else:
        return []
def extract_info_from_one_subdir(root_dir):
    # List immediate subdirectories
    immediate_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not immediate_subdirs:
        print("No subdirectories found.")
        return None, None
    
    # Choose the first subdirectory
    subdir = immediate_subdirs[0]
    
    # Extract context: all numbers between "seed8_" and "_pb2"
    #context_match = re.search(r'seed\d+_(.*?)_pb2', subdir)
    #if context_match:
        # Find all numeric values within the matched context and store them in a list
    context_values =extract_context_from_dir_name(subdir)
    # else:
    #     context_values = None

    # The run ID is simply the name of the chosen subdirectory
    run_id = subdir
    
    return context_values, run_id



# Class definitions
class TrialMetadata:
    def __init__(self, trial_id: str, hyperparameters: Dict[str, Any], metrics: Dict[str, Any], time_step: int, context_feature: Any):
        self.trial_id = trial_id
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.time_step = time_step
        self.context_feature = context_feature

    def __repr__(self):
        return f"TrialMetadata(trial_id={self.trial_id}, time_step={self.time_step}, context_feature={self.context_feature})"

class MetadataCollection:
    def __init__(self):
        self.trials = []

    def add_trial(self, trial: TrialMetadata):
        self.trials.append(trial)

    def find_by_time_step(self, time_step: int) -> List[TrialMetadata]:
        Xraw = []
        yraw = []
        context=[]
        for trial in self.trials:
            boolean_series = trial.time_step == time_step
            indices = boolean_series[boolean_series].index

            Xraw.append(trial.hyperparameters.loc[indices].to_numpy())
            yraw.append(trial.metrics.loc[indices].to_numpy())
            context.append(trial.context_feature)
        return Xraw, yraw, context

            

    def find_before_time_step(self, max_time_step: int) -> List[TrialMetadata]:
        Xraw = np.array([])
        yraw = []
        context=[]
        for trial in self.trials:
            boolean_series = trial.time_step <=  max_time_step
            indices = boolean_series[boolean_series].index
            time = trial.time_step.loc[indices].values
            hps = trial.hyperparameters.loc[indices].values
            reward = trial.metrics.loc[indices].values    
            combined = np.hstack((time,reward, hps ))
            Xraw=np.append(Xraw, combined).reshape(-1,combined.shape[1])
            #Xraw.append(combined)
            yraw.extend(trial.metrics.diff().fillna(0).loc[indices].values)
            context.extend(trial.context_feature)
        return Xraw, yraw, context

    # def find_by_time_range(self, min_time_step: int, max_time_step: int) -> List[TrialMetadata]:
    #     return [trial for trial in self.trials if min_time_step <= trial.time_step <= max_time_step]

    def __repr__(self):
        return f"MetadataCollection(trials={len(self.trials)})"

class TrialLogProcessor:
    def __init__(self, job_log_dirs: List[str], hyperparameters_tuned: List[str], metric, time_attr):
        self.job_log_dirs = job_log_dirs
        self.hyperparameters_tuned = hyperparameters_tuned
        self.metadata_collection = MetadataCollection()
        self.metric = metric
        self.time_attr= time_attr

    def find_and_process_logs(self, context_feature=None):
        for job_dir in self.job_log_dirs:
            for dirpath, dirnames, filenames in os.walk(job_dir):
                if 'progress.csv' in filenames:
                    trial_log_path= os.path.join(dirpath, 'progress.csv')
                    explicit_context_feature, trial_id = extract_info_from_one_subdir(job_dir)
                    if context_feature is None:
                        context_feature=explicit_context_feature
                    self.process_log_file(trial_log_path, trial_id=trial_id, context_feature=context_feature)


    def process_log_file(self, trial_log_path: str, trial_id: str, context_feature: Any):

        trial_progress = pd.read_csv(trial_log_path)
        time_steps = trial_progress[[self.time_attr]]
        metric= trial_progress[[self.metric]]
        hyperparameters_tuned = trial_progress.loc[:, self.hyperparameters_tuned]
        trial_metadata = TrialMetadata(
                    trial_id=trial_id,
                    hyperparameters=hyperparameters_tuned,
                    metrics=metric,
                    time_step=time_steps,
                    context_feature=context_feature
                )

        self.metadata_collection.add_trial(trial_metadata)

    def get_metadata_collection(self) -> MetadataCollection:
        return self.metadata_collection

# Main function to handle command-line arguments and execute the script



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process trial logs from job directories.")
    parser.add_argument('job_log_dirs', type=str, nargs='+', help="Paths to job log directories")
    parser.add_argument('--hyperparameters_tuned', type=str, nargs='+', default=['info/learner/default_policy/learner_stats/cur_lr'], help="List of hyperparameters to extract")
    parser.add_argument('--time_attr', type=str, default='num_env_steps_trained')
    parser.add_argument('--metric', type=str, default='env_runners/episode_reward_mean')

    path='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/logs_pb2_dkl_mcd_c10/2024-09-03_PPO_gravity_0.0025_pb2_dkl_Size8_CARLMountainCar_timesteps_total/pb2_dkl_CARLMountainCar_seed8_gravity_0.0025/PPO_CARLMountainCar_6dd38_00007_7_2024-09-03_02-14-04/progress.csv'
    df=pd.read_csv(path)
    #print(df.columns)
    #print(df['info/learner/default_policy/learner_stats/cur_lr'])
    args = parser.parse_args()
    # print(args.job_log_dirs)

    processor = TrialLogProcessor(job_log_dirs=args.job_log_dirs, hyperparameters_tuned=args.hyperparameters_tuned, metric=args.metric, time_attr=args.time_attr)
    processor.find_and_process_logs()

    metadata_collection = processor.get_metadata_collection()

    trials_at_time_step_4000 = metadata_collection.find_before_time_step(4000)

    # combined_array = np.vstack(trials_at_time_step_4000)
    # print(combined_array.shape)

    print(np.array(trials_at_time_step_4000[0]).shape)

    # print(trials_in_time_range)
    #s =eval("{'env': 'CARLMountainCar', 'log_level': 'INFO', 'seed': 9, 'kl_coeff': 1.0, 'num_gpus': 0, 'horizon': 1000, 'observation_filter': 'MeanStdFilter', 'model': {'fcnet_hiddens': [32, 32]}, 'lr': 0.0005929241830539595, 'env_config': {'gravity': 0.00025}}")

    
    # Replace single quotes with double quotes to ensure proper JSON format
    #s = s.replace("'", '"')

