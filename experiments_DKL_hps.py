import random

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import pearsonr
import ray
from ray import train, tune
from ray.tune.utils.util import flatten_dict
import math
import argparse
import json
import os
import random
from datetime import datetim e
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import GPy
from DKL.pb2_utils import (
        
        TV_SquaredExp,
        get_limits,
        
    )

from DKL.pb2_utils import (
        UCB,
        normalize,
        optimize_acq,
        select_length,
        standardize,
    )

import torch
#from DKL.pretraining_dkl import PB2_dkl_pretrained#, GPRegressionModel_DKL
from DKL.Meta_DKL_train_utils import UCB_DKL, metatrain_DKL, metatrain_DKL_wilson,  GPRegressionModel_DKL, GPRegressionModel_DKL_1, optimize_acq_DKL
from DKL.Meta_DKL_data_utils import process_pb2_runs_metadata

from copy import deepcopy

import gpytorch
import torch
import torch.nn as nn
import random
import numpy as np
from ray import train
import re 



def extract_params_from_path(path):
    # Define a regular expression to extract the part between the environment name and 'pb2'
    # Step 1: Extract the substring between the first underscore and _pb2
    directories = path.split('/')
    # Extract the second-to-last directory
    if len(directories) > 1:
        exp_name = directories[-2]  # Get the second-to-last element
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


def dynamic_exp(config):
    train_dir_list = config['train_dir_list']
    test_dir = config['test_dir']

    seed = config['seed']
    test_env_config = extract_params_from_path(test_dir)
    #x_train, y_train, x_val, y_val = process_metadata(meta_data_dir_list=train_dir_list, hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config))
    X, y =  process_pb2_runs_metadata(meta_data_dir_list=train_dir_list, hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config), partition_val=False)
    x_test, y_test = process_pb2_runs_metadata([test_dir], hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config), partition_val=False)
    # parttioton by t_perutb
    x_test = x_test.values
    y_test = y_test.values
    t_perturb_simulation  12 # 50000 / 4000 = 12 if u want to test for one PERTURB interv
    perf= []
    try:
        neural_network = create_nn_model(config)
        for i in range(t_perturb_simulation, len(x_test), t_perturb_simulation):
            Xraw = x_test[:i, :]
            yraw = y_test[:i]

            x_test_exp = x_test[i:, :]
            y_test_exp = y_test[i:]     
            
            
            rsme_meta_model = create_and_test_meta_model(Xraw, yraw, config, neural_network, X, y, x_test_exp, y_test_exp)
            rsme_baseline = create_and_test_baseline_model(Xraw, yraw, config, x_test_exp, y_test_exp)
            

            #train.report({'rsme_meta': rsme_meta_model, 'rsme_baseline': rsme_baseline, 'is_better': rsme_baseline > rsme_meta_model, 'iteration': i})
            
            print(f'{i}th interval done')
            if rsme_baseline < rsme_meta_model:
                print('rabena yestor')
            else:
                print('rabena satar')
            perf.append({'pb2': rsme_baseline, 'meta':rsme_meta_model})
        perf_df = pd.DataFrame(data=perf)
        perf_df.to_csv(f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/RSME/rsme.csv')
    except Exception as e:
        print(e)
        perf_df = pd.DataFrame(data=perf)
        perf_df.to_csv(f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/RSME/rsme.csv')





   


def create_gp(config):
    train_dir_list = config['train_dir_list']
    test_dir = config['test_dir']
    print(test_dir)
    seed = config['seed']
    lr = config['lr']
    test_env_config = extract_params_from_path(test_dir)
    x_test, y_test = process_metadata([test_dir], hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config), partition_val=False)
    #print(x_test.describe())
    
    model = create_nn_model(config)
    # pretrain
    num_epochs_nn = config['num_epochs_nn']        
    x_train, y_train, x_val, y_val = process_metadata(meta_data_dir_list=train_dir_list, hyperparams_bounds=config['hyperparam_bounds'], current_env_config=deepcopy(test_env_config))
    print(y_train.describe(), y_val.describe())
    # for col in x_train.columns:  # Assuming you are using pandas DataFrames
    #     x_train_vis = x_train
    #     x_train_vis['index'] = np.arange(0, len(x_train))
    #     x_train_vis.set_index('index')
    #     x_val_vis = x_val
    #     x_val_vis['index'] = np.arange(0, len(x_val_vis))
    #     x_val_vis.set_index('index')

    #     sns.kdeplot(x_train_vis[col], label='Train')
    #     sns.kdeplot(x_val_vis[col], label='Validation')
    #     plt.title(f"Distribution of {col}")
    #     plt.legend()
    #     plt.savefig(f"Distribution of {col}.png")
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler_train.fit_transform(x_train)
    x_val = scaler_train.transform(x_val)
    model = pretrain_neural_network_model_with_sched(model=model, X_train=x_train,y_train=y_train, X_val=x_val,y_val= y_val ,seed=seed, num_epochs=num_epochs_nn)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # shouldnt you combine train and val here? for the gp?
    
    train_x_gp = torch.tensor(x_train, dtype=torch.float32)
    train_y_gp =torch.squeeze(torch.tensor(y_train, dtype=torch.float32))
    gp = GPRegressionModel_DKL(train_x_gp, train_y_gp,  likelihood, seed,model, False )
    val_x_gp = torch.tensor(x_val, dtype=torch.float32)
    val_y_gp =torch.squeeze(torch.tensor(y_val, dtype=torch.float32))   
    num_epochs_gp = config['num_epochs_gp']

    m_trained, mll_m ,l= metatrain_DKL_wilson(model=gp, X_train=train_x_gp, y_train=train_y_gp, X_test=val_x_gp, y_test=val_y_gp, likelihood=likelihood,seed=seed, ray_tune_exp=True, lr=lr, training_iterations=num_epochs_gp)

    # evaluation phase

    m_trained.eval()
    l.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():

        x_test = scaler_train.transform(x_test)
        test_x_gp = torch.tensor(x_test, dtype=torch.float32)
        test_y_gp =torch.squeeze(torch.tensor(y_test, dtype=torch.float32))
        preds = m_trained(test_x_gp)
        rsme_test = torch.sqrt(torch.mean((preds.mean - test_y_gp) ** 2)).item()
        print('RSME TEST ', rsme_test)
    train.report({'rsme_test':rsme_test})
    #test_baseline_model(x_train, y_train,x_test,y_test )




def create_and_test_meta_model(Xraw, yraw, config, neural_network, meta_x, meta_y, x_test, y_test):
    hp_bounds = config['hyperparam_bounds']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    bounds = flatten_dict(
            hp_bounds, prevent_delimiter=True
        )    
    
    seed = config['seed']

    if len(Xraw) < 200:
        # val set contains meta data 
        X_train_split, X_val, y_train_split, y_val = train_test_split(
        meta_x.values, meta_y.values, 
        test_size=0.1,  # 20% for validation
        random_state=seed  # Set random seed for reproducibility
        )
        
        X_train = np.concatenate([X_train_split, Xraw], axis=0)
        
        scaler_rewards= MinMaxScaler()
        yraw = scaler_rewards.fit_transform(yraw.reshape(-1, 1)) 
        y = np.concatenate([y_train_split, yraw.flatten() ], axis=0)
        
    else:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
        Xraw, yraw, 
        test_size=0.2,  # 20% for validation
        random_state=seed  # Set random seed for reproducibility
        )
        
        X_train = np.concatenate([meta_x.values, X_train_split], axis=0)
        scaler_rewards= MinMaxScaler()
        y_train_split = scaler_rewards.fit_transform(y_train_split.reshape(-1, 1)) 
        y_val = scaler_rewards.transform(y_val.reshape(-1, 1))
    
        y = np.concatenate([meta_y.values, y_train_split.flatten()], axis=0)
    oldpoints = X_train[:, :3 ]

    limits = get_limits(oldpoints, bounds)
    X = normalize(X_train, limits)
    
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape[0],-1)
    y_val = scaler.transform(y_val.reshape(-1,1).reshape(y_val.shape[0], -1))
    
    print(pd.DataFrame(data=y_val).describe(), 'DESCIBE')
    train_x = torch.tensor(X, dtype=torch.float32)
    train_y =torch.squeeze(torch.tensor(y, dtype=torch.float32))

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val =torch.squeeze(torch.tensor(y_val, dtype=torch.float32))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()


    m = GPRegressionModel_DKL(train_x, train_y, feature_extractor=neural_network, likelihood=likelihood,seed=seed)
    m_trained, _ ,l= metatrain_DKL(model=m, X_train=train_x, y_train=train_y, likelihood=likelihood,seed=seed, X_val=X_val, y_val=y_val)

    x_test = normalize(x_test, limits)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    print('going to eval')
    m_trained.eval()
    l.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = m_trained(x_test)
    
    y_test = scaler_rewards.transform(y_test.reshape(-1, 1))
    rmse = root_mean_squared_error(y_test, np.array(scaler.inverse_transform(preds.mean.reshape(-1, 1))))#torch.sqrt(torch.mean((preds.mean - y_test)**2))
    rmse = rmse/math.sqrt(len(x_test))
    print(pd.DataFrame(data = preds.variance, columns = ['metapreds']).describe())
    print(f'RSME META MODEL: {rmse}') 
    print('correlation', np.corrcoef(preds.mean, preds.variance), preds.mean.shape)
    optimize_acq_DKL(UCB_DKL, m_trained, m_trained,l,l, Xraw[-1,:3], 3 ,seed)
    
    return rmse


def create_and_test_baseline_model(Xraw, yraw, config, x_test, y_test):
    

    # removing similariyt featurws
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    hp_bounds = config['hyperparam_bounds']
    # delete the similarity feature
    num_f = 2
    Xraw = np.delete(Xraw, 2, axis=1)
    x_test = np.delete(x_test, 2, axis=1) 
    
    bounds = flatten_dict(
            hp_bounds, prevent_delimiter=True
        )

    length = select_length(Xraw, yraw, bounds, num_f)

    Xraw = Xraw[-length:, :]
    yraw = yraw[-length:]

    base_vals = np.array(list(bounds.values())).T
    oldpoints = Xraw[:, :num_f]
    # old_lims = np.concatenate(
    #     (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    # ).reshape(2, oldpoints.shape[1])
    # limits = np.concatenate((old_lims, base_vals), axis=1)
    limits = get_limits(oldpoints, bounds)

    X = normalize(Xraw, limits)
    scaler = StandardScaler()

    y = scaler.fit_transform(yraw.reshape(-1, 1) )

    x_test = normalize(x_test, limits)
    kernel = TV_SquaredExp(
        input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
    )

    try:
        m = GPy.models.GPRegression(X, y, kernel)
    except np.linalg.LinAlgError:
        # add diagonal ** we would ideally make this something more robust...
        X += np.eye(X.shape[0]) * 1e-3
        m = GPy.models.GPRegression(X, y, kernel)

    try:
        m.optimize()
    except np.linalg.LinAlgError:
        # add diagonal ** we would ideally make this something more robust...
        X += np.eye(X.shape[0]) * 1e-3
        m = GPy.models.GPRegression(X, y, kernel)
        m.optimize()

    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))
    
    preds, pred_vars = m.predict(x_test)
    
    y_test = scaler.transform(y_test.reshape(-1, 1))
    # Step 3: Calculate RMSE
    rmse = root_mean_squared_error(y_test, scaler.inverse_transform(preds.reshape(-1, 1)))/math.sqrt(len(x_test))
    print(pd.DataFrame(data = pred_vars, columns = ['basepreds']).describe())
    print(f"RMSE BASELINE: {rmse}")
    print('baslinecorr', np.corrcoef(preds.reshape(-1), pred_vars.reshape(-1)))
    

    #corr, _ = pearsonr(preds.reshape(-1), pred_vars.reshape(-1))
    #print(corr)
    #optimize_acq(UCB, m, m,  Xraw[-1,:3], num_f,0)
    return rmse

    


def plot(x_train, y_train, x_test, y_pred_mean,  y_pred_var):
    import matplotlib.pyplot as plt

# Assume x_test is 1D for simple plotting, reshape if necessary
    plt.figure(figsize=(10, 6))

    # Plot the training data
    plt.scatter(x_train, y_train, color='blue', label='Training Data')

    # Plot the predicted mean
    plt.plot(x_test, y_pred_mean, color='red', label='Predicted Mean')

    # Plot the uncertainty (confidence intervals)
    plt.fill_between(
        x_test.flatten(),  # Assuming x_test is 1D, otherwise adjust accordingly
        (y_pred_mean - 1.96 * np.sqrt(y_pred_var)).flatten(),
        (y_pred_mean + 1.96 * np.sqrt(y_pred_var)).flatten(),
        color='gray', alpha=0.3, label='Confidence Interval (95%)'
    )

    plt.legend()
    plt.title('GP Predictions with Confidence Intervals')
    plt.savefig('GP Predictions with Confidence Intervals.png')

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
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 50},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 2}  # Feature extraction output size
]

archi_4 = [
    {'type': 'linear', 'out_dim': 1000},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 500},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 450},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 350},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 250},
    {'type': 'relu'},
    {'type': 'linear', 'out_dim': 100},
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
path_5='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.30_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c20/CARLBipedalWalker_TERRAIN_LENGTH_400.0_pb2_Size_8_timesteps_total/seed3'
path_6 = '/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.31_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c9/CARLBipedalWalker_TERRAIN_LENGTH_180.0_pb2_Size_8_timesteps_total/seed5'
path_7 ='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-10-14_16:40.23_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c17/CARLBipedalWalker_TERRAIN_LENGTH_340.0_pb2_Size_8_timesteps_total/seed0'
path_8='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-10-14_16:40.24_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c19/CARLBipedalWalker_TERRAIN_LENGTH_380.0_pb2_Size_8_timesteps_total/seed0'
search_space = {
    #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
    'train_dir_list' : tune.grid_search([[path_1],[path_2,path_1, path_3]]),
    'test_dir' : tune.grid_search([path_1,path_4]),
    'seed' : 0,
    'lr' : 0.01, #tune.grid_search([0.01, 0.001, 0.0001]), # lr - gp #
    'num_epochs_nn': 5, #tune.grid_search([100]),
    'hyperparam_bounds': {
                "lambda_": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                #'gamma': [0.9,0.99],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 10_000],
                'num_sgd_iter': [3,30]
             },
    'num_epochs_gp': 5,
    'architecture': tune.grid_search([archi_1, archi_2, archi_3])

}
search_space = {
    #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
    'train_dir_list' : tune.grid_search([[path_1,path_2,path_4 ]]),
    'test_dir' : tune.grid_search([path_3]),
    'seed' : 0,
    'lr' : tune.grid_search([0.0001]), # lr - gp #
    'num_epochs_nn': tune.grid_search([3]),
    'hyperparam_bounds': {
                "lambda": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                #'gamma': [0.9,0.99],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 10_000],
                'num_sgd_iter': [3,30]
             },
    'num_epochs_gp': tune.grid_search([100]),
    'architecture': tune.grid_search([archi_1])

}
sampled_config = {
    'train_dir_list': [path_7],  # Only one option
    'test_dir': path_8,                         # Only one option
    'seed': 0,                                  # Fixed value
    'lr': 0.01,                                 # Only one option from grid search
    'num_epochs_nn': 250,                         # Only one option
    'hyperparam_bounds': {
                "lambda": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                #'gamma': [0.9,0.99],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 10_000],
                'num_sgd_iter': [3,30]
             },
    'num_epochs_gp': 40,                       # Only one option
    'architecture': archi_1       # Only one option
}

def complex_function(X):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    # Create a complex relationship combining trigonometric, exponential, and polynomial terms
    y = np.sin(2 * np.pi * x1) + np.cos(3 * np.pi * x2) + np.exp(0.5 * x3) + 0.1 * x1**2 - 0.05 * x2**3
    return y

def generate_dummy_data():
    
    from math import floor
    n_samples = 2000
    np.random.seed(0)
    X = np.random.uniform(-5, 5, size=(n_samples, 7))
# #X = torch.randn(2000, 3)
    y = complex_function(X)
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :]
    train_y = y[:train_n].flatten()
    X_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(X[train_n:, :], dtype=torch.float32)
    test_y = torch.tensor(y[train_n:], dtype=torch.float32)
    return X_train, y_train, test_x, test_y
dynamic_exp(config=sampled_config)


