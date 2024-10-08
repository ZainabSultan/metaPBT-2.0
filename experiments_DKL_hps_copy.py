import random

import ray
from ray import train, tune

import argparse
import json
import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import torch
from DKL.pretraining_dkl import PB2_dkl_pretrained, GPRegressionModel_DKL
from DKL.pretraining_dkl_utils import metatrain_DKL_wilson, pretrain_neural_network_model_with_sched, process_metadata

from copy import deepcopy

import gpytorch
import torch
import torch.nn as nn
import random
import numpy as np
from ray import train
import re 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000_000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--t_ready", type=int, default=500_00)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=str, default='{"gravity":1.0}')
    parser.add_argument(
        "--horizon", type=int, default=1600
    )  # make this 1000 for other envs
    parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
    parser.add_argument("--env_name", type=str, default="CARLMountainCar") #"CartPole-v1"
    parser.add_argument(
        "--criteria", type=str, default="timesteps_total"
    )  # "training_iteration", "time_total_s"
    parser.add_argument(
        "--net", type=str, default="32_32"
    )  # May be important to use a larger network for bigger tasks.
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--scheduler", type=str, default="metadkl")  # ['pbt', 'pb2']
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument('--ws_dir', type=str, default=r'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir')
    parser.add_argument('--metric', type=str, default='env_runners/episode_reward_mean')
    #parser.add_argument('--metadata_dir_list', nargs='+', help='List of directories', required=True)

    args = parser.parse_args()



def extract_params_from_path(path):
    # Define a regular expression to extract the part between the environment name and 'pb2'
    # Step 1: Extract the substring between the first underscore and _pb2
    directories = path.split('/')
    # Extract the second-to-last directory
    if len(directories) > 1:
        exp_name = directories[-2]  # Get the second-to-last element
        print(exp_name)
        env_name = exp_name.split('_')[0]
        match = re.search(r'_(.*?)_pb2', exp_name)
        if match:
            extracted_part = match.group(1)  # Get the matched part
        else:
            extracted_part = ""

        # Step 2: Create a dictionary from the extracted part
        if extracted_part:
            # Split by underscores
            
            pairs = extracted_part.split('_')
            print(pairs)
            config = {'env_name': env_name}
            feature_name = ''
            for i in pairs:
                try:
                    i = float(i)
                    config[feature_name] = float(i)
                    feature_name = ''

                except:
                    if feature_name == '':
                        feature_name+=i
                    else:
                        feature_name+='_'
                        feature_name+=i
               
                    

        # Print the result dictionary
        print(config)
        return config



class DynamicNN(nn.Module):
    def __init__(self, architecture, input_dim, seed, joint_gp_training_phase=False):
        super(DynamicNN, self).__init__()
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.joint_gp_training_phase = joint_gp_training_phase
        self.layers = nn.ModuleList()  # Use ModuleList to dynamically append layers
        
        prev_dim = input_dim
        for layer_info in architecture:
            layer_type = layer_info['type']
            
            if layer_type == 'linear':
                out_dim = layer_info['out_dim']
                self.layers.append(nn.Linear(prev_dim, out_dim))
                prev_dim = out_dim
            
            elif layer_type == 'relu':
                self.layers.append(nn.ReLU())
            
            # Add more types as needed (e.g., dropout, batchnorm)
        
        # Define a final output layer if needed
        self.final_layer = nn.Linear(prev_dim, 1)  # Output size 1 for regression or binary classification

    def extract(self, x):
        # Pass input through the layers (up to the last one before the final output)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward(self, x):
        x = self.extract(x)
        if not self.joint_gp_training_phase:
            x = self.final_layer(x)  # Apply the final output layer in some training phases
        return x


# Example usage:
# Architecture is a list of layers with their types and other configurations
architecture = [
    {'type': 'linear', 'out_dim': 1000},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
]

# Dynamically create the neural network
#input_dim = 10  # Example input dimension
#model = DynamicNN(architecture, input_dim, seed=42)

# Print the architecture of the dynamically created model
#print(model)
# pretrain the model ?
def create_gp(config):
    train_dir_list = config['train_dir_list']
    test_dir = config['test_dir']
    print(test_dir)
    seed = config['seed']
    lr = config['lr']
    test_env_config = extract_params_from_path(test_dir)
    x_test, y_test = process_metadata([test_dir], hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config), partition_val=False)
    
    model = create_nn_model(config)
    # pretrain
    num_epochs_nn = config['num_epochs_nn']        
    x_train, y_train, x_val, y_val = process_metadata(meta_data_dir_list=train_dir_list, hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config))
    model_trained = pretrain_neural_network_model_with_sched(model=model, X_train=x_train,y_train=y_train, X_val=x_val,y_val= y_val ,seed=seed, num_epochs=num_epochs_nn)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # shouldnt you combine train and val here? for the gp?
    
    train_x_gp = torch.tensor(x_train.values, dtype=torch.float32)
    train_y_gp =torch.squeeze(torch.tensor(y_train.values, dtype=torch.float32))
    gp = GPRegressionModel_DKL(train_x_gp, train_y_gp,  likelihood, seed,model_trained, False )
    val_x_gp = torch.tensor(x_val.values, dtype=torch.float32)
    val_y_gp =torch.squeeze(torch.tensor(y_val.values, dtype=torch.float32))   
    num_epochs_gp = config['num_epochs_gp']
    m_trained, mll_m ,l= metatrain_DKL_wilson(model=gp, X_train=train_x_gp, y_train=train_y_gp, X_test=val_x_gp, y_test=val_y_gp, likelihood=likelihood,seed=seed, ray_tune_exp=True, lr=lr, training_iterations=num_epochs_gp)

    # evaluation phase

    m_trained.eval()
    l.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        test_x_gp = torch.tensor(x_test.values, dtype=torch.float32)
        test_y_gp =torch.squeeze(torch.tensor(y_test.values, dtype=torch.float32))
        preds = m_trained(test_x_gp)
        rsme_test = torch.sqrt(torch.mean((preds.mean - test_y_gp) ** 2)).item()
        print('RSME TEST ', rsme_test)
    train.report({'rsme_test':rsme_test})
    





def create_nn_model(config):

    architecture = config['architecture']
    seed = config['seed']
    input_dim = len(list(config['hyperparam_bounds'].keys())) + 3
    model = DynamicNN(architecture, input_dim, seed=seed)
    return model


archi_1 =  [
    {'type': 'linear', 'out_dim': 1000},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
]

archi_2 = [
    {'type': 'linear', 'out_dim': 1000},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
]

archi_3 = [
    {'type': 'linear', 'out_dim': 1000},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
]

path_1='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.32_pb2_job_scripts.CARLBipedalWalker.TERRAIN_STEP.c18/CARLBipedalWalker_TERRAIN_STEP_0.8400000000000001_pb2_Size_8_timesteps_total/seed1'
path_2='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.31_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c4/CARLBipedalWalker_TERRAIN_LENGTH_80.0_pb2_Size_8_timesteps_total/seed2'
path_3='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.26_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c10/CARLBipedalWalker_TERRAIN_LENGTH_200.0_pb2_Size_8_timesteps_total/seed0'
path_4='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.31_pb2_job_scripts.CARLBipedalWalker.TERRAIN_STEP.c10/CARLBipedalWalker_TERRAIN_STEP_0.4666666666666667_pb2_Size_8_timesteps_total/seed9'

search_space = {
    #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
    'train_dir_list' : tune.grid_search([[path_1],[path_2,path_1, path_3]]),
    'test_dir' : tune.grid_search([path_1,path_4]),
    'seed' : 0,
    'lr' : 0.01, #tune.grid_search([0.01, 0.001, 0.0001]), # lr - gp #
    'num_epochs_nn': 100, #tune.grid_search([100]),
        'hyperparam_bounds': {
                "lambda": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                #'gamma': [0.9,0.99],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 10_000],
                'num_sgd_iter': [3,30]
             },
    'num_epochs_gp': 100,
    'architecture': tune.grid_search([archi_1, archi_2, archi_3])

}
# search_space = {
#     #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
#     'train_dir_list' : tune.grid_search([[path_1,path_2]]),
#     'test_dir' : tune.grid_search([path_3]),
#     'seed' : 0,
#     'lr' : tune.grid_search([0.01]), # lr - gp #
#     'num_epochs_nn': tune.grid_search([3]),
#     'hyperparam_bounds': {
#                 "lambda": [0.9, 0.99],
#                 "clip_param": [0.1, 0.5],
#                 #'gamma': [0.9,0.99],
#                 "lr": [1e-5, 1e-3],
#                 #"train_batch_size": [1000, 10_000],
#                 'num_sgd_iter': [3,30]
#              },
#     'num_epochs_gp': tune.grid_search([2]),
#     'architecture': tune.grid_search([archi_1])

# }

ray.init()  
tune.run(create_gp, config=search_space, num_samples=2, name='testing_report', storage_path=args.ws_dir)