


import numpy as np

import torch
from ray.tune.utils.util import flatten_dict
import os
import json
from sklearn.preprocessing import StandardScaler
import pandas as pd
import gpytorch

import torch
import logging

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

import numpy as np


def load_json_file(file_path):
    """
    Loads a JSON file where each line is a separate JSON object and returns it as a list of dictionaries.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
##ä META DATA UTILS

def process_pb2_runs_metadata(meta_data_dir_list, hyperparams_bounds, similarity_feature_list=[],current_env_config={}, time_attr='timesteps_total', metric='episode_reward_mean', best=False, partition_val=True):
    current_env_config.pop('env_name')
    hps_list = list(hyperparams_bounds.keys())
    df_list = []
    _hyperparam_bounds_flat = flatten_dict(
            hyperparams_bounds, prevent_delimiter=True
        )
    len_each_run = []
    for meta_data_dir in meta_data_dir_list:
        if best:
            ... # todo extract best schedule
        else:
            # learn from all
            for sample_dir in os.listdir(meta_data_dir):
                if os.path.isdir(os.path.join(meta_data_dir,sample_dir)):
                    sample_full_path= os.path.join(meta_data_dir,sample_dir)
                    result_json_file = os.path.join(sample_full_path, 'result.json')
                    results = load_json_file(result_json_file)
                    if similarity_feature_list == [] and current_env_config!={}:
                        context_features=results[0]['config']['env_config']
                        env_name = context_features.pop('env_name')
                        default_context= get_default_context(env_name)
                        for key, default in default_context.items():
                            context_features.setdefault(key, default)
                            current_env_config.setdefault(key, default)
                        print(current_env_config, context_features)
                        current_env_context_values = np.array(list(current_env_config.values()))
                        context_values = np.array(list(context_features.values()))
                        similarity = current_env_context_values - context_values
                        effective_similiayrity = similarity[similarity != 0]
                        if len(effective_similiayrity)==0: # from the same env, sim is 0
                            effective_similiayrity=0.0 
                        elif len(effective_similiayrity)>=1:
                            effective_similiayrity=effective_similiayrity[0]
                        similarity_feature ={'similiarity_feature': effective_similiayrity}
                        
                    data=[]
                    for result_row in results:
                        extracted_values = {key: result_row['config'][key] for key in hps_list if key in results[0]['config']}
                        reward ={metric: result_row['env_runners'][metric]} 
                        time = {time_attr:  result_row[time_attr]}
                        row = time| reward |  extracted_values |similarity_feature
                        data.append(row)
                    # normalise and reward changes per worker
                    data = pd.DataFrame(data=data)
                    data['reward_changes'] = data[metric].diff().fillna(0)
                    fixed_colnames = [time_attr, metric, 'similiarity_feature' ] # fixed
                    fixed_colnames.extend(hps_list) 
                    fixed_colnames.extend(['reward_changes'])
                    data = data[fixed_colnames] #reordering cols  to fit pb2 existing code            
                    X = data.drop(columns=['reward_changes'])#normalize(data.drop(columns=['reward_changes','similiarity_feature']), limits)
                    reward_scaler = StandardScaler()
                    rewards_scaled = reward_scaler.fit_transform(pd.DataFrame(data=data['reward_changes']))
                    y_df = pd.DataFrame(rewards_scaled, columns=['reward_changes'], index=X.index)
                    result = pd.concat([X, y_df], axis=1)
                    df_list.append(result)
            len_each_run.append(len(df_list))
                    
    if partition_val:
        # Subtract 1 from each index (to match Python indexing, which is 0-based)
        adjusted_indices = [i - 1 for i in len_each_run] # because i want to take 1 seed from every run!
        
        # Extract the selected DataFrames into one list
        metdata_val_df_list = [df_list[i] for i in adjusted_indices]
        # Extract the remaining DataFrames into another list
        metdata_train_df_list = [df for idx, df in enumerate(df_list) if idx not in adjusted_indices]
        # Combine the selected DataFrames into one DataFrame
        metdata_train_df = pd.concat(metdata_train_df_list, ignore_index=True)
        metdata_val_df = pd.concat(metdata_val_df_list, ignore_index=True)
        # Combine the remaining DataFrames into another DataFrame
        x_train = metdata_train_df.drop(columns=['reward_changes'])
        y_train = metdata_train_df['reward_changes']
        x_val = metdata_val_df.drop(columns=['reward_changes'])
        y_val = metdata_val_df['reward_changes']
        return x_train, y_train, x_val, y_val
    else:
        # Combine training and validation sets
        metdata_df = pd.concat(df_list, ignore_index=True)
        X= metdata_df.drop(columns=['reward_changes'])
        y = metdata_df['reward_changes']
        return X, y

def get_default_context(env_name):
    env_name = env_name.lower()
    if 'cartpole' in env_name:
        return {'length': 0.5, 'tau':0.02, 'gravity':9.8}
    elif 'mountaincar' in env_name:
        return {'gravity': 0.0025}
    elif 'bipedal' in env_name:
        return {
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
    else:
        raise NotImplementedError()
    
def normalize(data, wrt):
    """Normalize data to be in range (0,1), with respect to (wrt) boundaries,
    which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (
        np.max(wrt, axis=0) - np.min(wrt, axis=0) + 1e-8
    )


def standardize(data):
    """Standardize to be Gaussian N(0,1). Clip final values."""
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return data

    
