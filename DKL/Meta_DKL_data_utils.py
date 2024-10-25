


import numpy as np

import torch
from ray.tune.utils.util import flatten_dict
import os
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import gpytorch
from context_mapper import get_defaults, get_context_id
import torch
import logging

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
import copy
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
    if 'lambda_' in hps_list:
        lambda_index = hps_list.index('lambda_')  # Find the index of 'banana'
        hps_list[lambda_index] = 'lambda'  
    df_list = []

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
                        default_context= get_defaults(env_name)
                        for key, default in default_context.items():
                            context_features.setdefault(key, default)
                            current_env_config.setdefault(key, default)
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
                    reward_scaler = MinMaxScaler()
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


    
def normalize(data, wrt):
    """Normalize data to be in range (0,1), with respect to (wrt) boundaries,
    which can be specified.
    """
    return (data - np.min(wrt, axis=0)) / (
        np.max(wrt, axis=0) - np.min(wrt, axis=0) + 1e-8
    )


def standardize(data):
    """Standardize to be Gaussian N(0,1)"""
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return data


def get_metadata_dirs(env_config, num_meta_contexts, method='closest', exps_dir=''):
    if exps_dir=='':
        exps_dir ='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/CARLBipedalWalker_4_agents'
    print(exps_dir)
    
    env_config_copy = copy.deepcopy(env_config)
    env_name = env_config_copy.pop('env_name')
    feature = next(iter(env_config_copy.keys()))
    id = get_context_id(env_name, env_config_copy)
    id_list = np.arange(1,22)
    
    sorted_ids = sorted(id_list, key=lambda id_val: abs(id_val - id))
    if method == 'closest':
        closest_ids = sorted_ids[:num_meta_contexts]
        target_envs = [f'{env_name}.{feature}.c{i}' for i in closest_ids]
        target_envs.extend([f'{env_name}_{feature}_context_{i}' for i in closest_ids]) # alternative name


    meta_dirs = []
    for root, dirs, files in os.walk(exps_dir):
        for dir_name in dirs:
            if any(substring in dir_name for substring in target_envs):#if dir_name in target_envs:
                if 'meta' not in dir_name: # to eliminate meta runs
                    first_level_dir = os.path.join(root, dir_name)
                    
                    # Step 2: Get the only subdirectory inside the found directory
                    second_level_subdirs = next(os.walk(first_level_dir))[1]
                    if len(second_level_subdirs) == 1:
                        second_level_dir = os.path.join(first_level_dir, second_level_subdirs[0])
                        
                        # Step 3: Find the subdirectory named 'seed0'
                        seed0_dir = os.path.join(second_level_dir, 'seed0')
                        if os.path.exists(seed0_dir) and 'old' not in seed0_dir:
                            meta_dirs.append(seed0_dir)
                        else:
                            print(f"'seed0' directory not found inside {second_level_dir} or its an old run")
                    else:
                        print(f"More than one subdirectory found in {first_level_dir}")
                
    #print(f"No directory with substring '{target_substring}' found.")
    return meta_dirs
    
    
# def select_length(Xraw, yraw, bounds, num_f):
#     """Select the number of datapoints to keep, using cross validation"""
#     min_len = 200

#     if Xraw.shape[0] < min_len:
#         return Xraw.shape[0]
#     else:
#         length = min_len - 10
#         scores = []
#         while length + 10 <= Xraw.shape[0]:
#             length += 10

#             base_vals = np.array(list(bounds.values())).T
#             X_len = Xraw[-length:, :]
#             y_len = yraw[-length:]
#             oldpoints = X_len[:, :num_f]
#             old_lims = np.concatenate(
#                 (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
#             ).reshape(2, oldpoints.shape[1])
#             limits = np.concatenate((old_lims, base_vals), axis=1)

#             X = normalize(X_len, limits)
#             y = standardize(y_len).reshape(y_len.size, 1)

#             kernel = TV_SquaredExp(
#                 input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
#             )
#             m = GPy.models.GPRegression(X, y, kernel)
#             m.optimize(messages=True)

#             scores.append(m.log_likelihood())
#         idx = np.argmax(scores)
#         print('score baseline: ', np.max(scores))
#         length = (idx + int((min_len / 10))) * 10
#         return length  
    # closest N envs
#m = get_metadata_dirs({'env_name':'CARLBipedalWalker', 'TERRAIN_LENGTH':360.0}, 2)
#print(m)
# df, y, v, vy = process_pb2_runs_metadata(['/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-10-14_16:40.24_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c19/CARLBipedalWalker_TERRAIN_LENGTH_380.0_pb2_Size_8_timesteps_total/seed0'
#                            ],
#                            current_env_config={'TERRAIN_LENGTH' : 380.0, 'env_name':'CARLBipedalWalker'},
#                             hyperparams_bounds=flatten_dict({
#             "lambda_": [0.9, 0.99],
#                 "clip_param": [0.1, 0.5],
#                 #'gamma': [0.9,0.99],
#                 "lr": [1e-5, 1e-3],
#                 #"train_batch_size": [1000, 10_000],
#                 'num_sgd_iter': [3,30]
#                 #'entropy_coeff' : [0.01, 0.5]
#             }, prevent_delimiter=True
#         )
                

#                            )

# scaler = StandardScaler()
# print(y)
# r = scaler.fit_transform(pd.DataFrame(data=y))
# print(pd.DataFrame(data=r).describe())
    






    

    


    
