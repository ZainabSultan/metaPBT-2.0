import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import spearmanr
from torch.optim import SGD, Adam
import torch
from ray.tune.utils.util import flatten_dict
import os
import json
from sklearn.preprocessing import StandardScaler, Normalizer
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import tqdm.notebook
import gpytorch
from DKL.pb2_utils import get_limits
from torch.optim.lr_scheduler import MultiStepLR
import torch
import logging
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
import random
import torch
import numpy as np
import random
import torch.nn as nn
import tracemalloc
from ray import train
from ray.tune.schedulers.pb2_utils import (

        normalize,

        #standardize,
        
    )

import matplotlib.pyplot as plt

def standardize(data):
    """Standardize to be Gaussian N(0,1). Clip final values."""
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return data

def optimize_acq_DKL(func, m, m1,l,l1, fixed, num_f, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    """Optimize acquisition function."""

    opts = {"maxiter": 200, "maxfun": 200, "disp": False}

    T = 10
    best_value = -999
    best_theta = m1.X[0, :]

    bounds = [(0, 1) for _ in range(m.X.shape[1] - num_f)]

    for ii in range(T):
        x0 = np.random.uniform(0, 1, m.X.shape[1] - num_f)

        res = minimize(
            lambda x: -func(m, m1,l,l1, x, fixed),
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options=opts,
        )

        val = func(m, m1,l,l1, res.x, fixed)
        if val > best_value:
            best_value = val
            best_theta = res.x

    return np.clip(best_theta, 0, 1)


def UCB_DKL(m, m1,l,l1, x, fixed, kappa=None):
    """UCB acquisition function. Interesting points to note:
    1) We concat with the fixed points, because we are not optimizing wrt
       these. This is the Reward and Time, which we can't change. We want
       to find the best hyperparameters *given* the reward and time.
    2) We use m to get the mean and m1 to get the variance. If we already
       have trials running, then m1 contains this information. This reduces
       the variance at points currently running, even if we don't have
       their label.
       Ref: https://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf

    """

    c1 = 0.2
    c2 = 0.4
    beta_t = c1 + max(0, np.log(c2 * m.X.shape[0]))
    kappa = np.sqrt(beta_t) if kappa is None else kappa

    xtest = np.concatenate((fixed.reshape(-1, 1), np.array(x).reshape(-1, 1))).T
    xtest = xtest.astype(np.float32)

    try:
        preds = predict(m, l, xtest)#m.predict(xtest)
        mean = preds.mean
        #mean = mean.astype(np.float32)
    except ValueError:
        logger.info('mean is error')
        print('value error in mean, defaulting to -9999')
        mean = -9999

    try:
        preds = predict(m1, l1, xtest)#m1.predict(xtest)
        var = preds.variance
        #var = preds.astype(np.float32)
    except ValueError:
        var = 0
        logger.info('error in varaince')
        print('value error in var, defaulting to 0')
    return mean + kappa * var

def predict(model, likelihood, x):

    model.eval()
    likelihood.eval()
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model(x)
    return preds

## KISS-GP wilson et al



import torch
import gpytorch
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging

logger = logging.getLogger(__name__)

def metatrain_DKL_wilson(model, X_train, y_train, likelihood, seed, 
                         training_iterations=100, freeze=False, 
                         warm_start_only=False, X_test=None, y_test=None, 
                         save=None, lr=0.01, ray_tune_exp=False, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, batch_size=64):
    
    

    tracemalloc.start()

    # Code block to monitor memory usage    

    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.train()
    likelihood.train()
    model.feature_extractor.joint_gp_training_phase=True

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=lr)

    if freeze or warm_start_only:
        # Don't train NN
        optimizer = torch.optim.Adam([
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    scheduler = scheduler(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_test is not None and y_test is not None:

        val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    logger.info('Training in progress!!')
    iterator = tqdm.tqdm(range(training_iterations))
    loss_values = []
    test_loss_vals = []

    
    for i in iterator:
        train_loss=0.0
        val_rmse=0.0
        val_nnl=0.0
        model.train()
        likelihood.train()
        #for batch_X, batch_y in train_loader:
        
            # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X_train)
        # Calculate loss and backprop derivatives
        loss = -mll(output, y_train)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        # Log the loss value
        train_loss += loss.item()
        train_loss /= len(train_loader)
        loss_values.append(train_loss)
        print('train_loss', train_loss)
        # Evaluate on the test set if provided
        if X_test is not None and y_test is not None:
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                for X_val, y_val in val_dataloader:
                    test_output = model(X_val)
                # Use the predictive distribution to get the mean and variance
                    pred_mean = test_output.mean
                    pred_var = test_output.variance
                # RMSE: Root Mean Squared Error
                    rmse = torch.sqrt(torch.mean((pred_mean - y_val) ** 2)).item() 
                # NLL: Negative Log Likelihood, sum or average over all test points
                    nll = -likelihood.log_marginal(y_val, test_output).mean().item()  # Take the mean NLL
                    val_rmse+=rmse
                    val_nnl+=nll
                
                val_rmse = val_rmse / len(val_dataloader)
                val_nll = val_nnl / len(val_dataloader)
                #scheduler.step(val_nll) #
                test_loss_vals.append(val_rmse)
                #print(f'RMSE: {rmse}, NLL: {nll}')
                if X_test is not None and y_test is not None:
                    print({'train_loss': train_loss, 'val_set_rmse': val_rmse, 'training_itr': i ,'negative_log_likelihood':val_nll})


            # if ray_tune_exp:
            #     train.report({'train_loss': train_loss, 'val_set_rmse': val_rmse, 'training_itr': i ,'negative_log_likelihood':val_nll})


    current, peak = tracemalloc.get_traced_memory()
    curr = round(current / (1024 ** 2) , 2)
    p = round(peak / (1024 ** 2), 2)

    print(f"Current memory usage: {curr} MB")
    print(f"Peak memory usage: {p} MB")
    
    tracemalloc.stop()
    #train.report({'peak_mem':p, 'current_mem': curr,'train_loss': train_loss, 'val_set_rmse': val_rmse, 'training_itr': i ,'negative_log_likelihood':val_nll})

    # Plot and save the training loss if specified
    # if save is not None:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(loss_values, label='Training Loss')
    #     plt.plot(test_loss_vals, label='val')
    #     plt.title('Training Loss over Iterations')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig('loss.png')
    #     plt.close()

    return model, mll, likelihood






def pretrain_neural_network_model_with_sched(model, X_train, y_train, X_val, y_val, seed, num_epochs=100, batch_size=64, learning_rate=1e-3, patience=10, scheduler_factor=0.1, scheduler_patience=5):
    # Ensure reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks

    # Learning rate scheduler (reduce LR when val loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True)

    # Prepare DataLoader for training and validation sets
    print(pd.DataFrame(data=X_train).describe())
    # scaler_train = StandardScaler()
    # X_train = scaler_train.fit_transform(X_train)
    print(pd.DataFrame(data=y_train).describe())
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    print(pd.DataFrame(data=y_val).describe())
    #X_val = scaler_train.transform(X_val)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.flatten(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_outputs = model(batch_X_val)
                val_loss += criterion(val_outputs.flatten(), batch_y_val).item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Step the scheduler based on validation loss
        #scheduler.step(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}. Restoring best model from epoch {epoch+1-patience} with val loss of {best_val_loss}")
            model.load_state_dict(best_model_state)  # Restore the best model
            break
    predictions = model(X_val).detach().cpu().numpy().flatten()
    residuals = y_val.numpy() - predictions
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.savefig('Residual Distribution.png')
    
    return model




def load_json_file(file_path):
    """
    Loads a JSON file where each line is a separate JSON object and returns it as a list of dictionaries.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def process_metadata(meta_data_dir_list, hyperparams_bounds, similarity_feature_list=[],current_env_config={}, time_attr='timesteps_total', metric='episode_reward_mean', best=False, partition_val=True):
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
                    # normalise and reward changes
                    data = pd.DataFrame(data=data)
                    #data['episode_reward_mean_norm'] = (data['episode_reward_mean'] - data['episode_reward_mean'].min()) / (data['episode_reward_mean'].max() - data['episode_reward_mean'].min())
                    data['reward_changes'] = data[metric].diff().fillna(0)
                    #data.drop(columns=[metric], inplace=True)
                    tr_colnames = [time_attr, metric ]
                    tr_colnames.extend(hps_list) 
                    tr_colnames.extend(['similiarity_feature', 'reward_changes'])
                    data = data[tr_colnames] #reordering cols
                    limits = get_limits(data[[time_attr, metric]], _hyperparam_bounds_flat)
                    
                    X = data.drop(columns=['reward_changes'])#normalize(data.drop(columns=['reward_changes','similiarity_feature']), limits)
                    # Add 'similiarity_feature' column back to X after the hyperparameters
                    #similarity_feature = standardize( data['similiarity_feature'])
                    #X = pd.concat([X, similarity_feature], axis=1)
                    #y = data['reward_changes'].values
                    y = standardize(data['reward_changes'].values).reshape(data['reward_changes'].size, 1)
                    y_df = pd.DataFrame(y, columns=['reward_changes'], index=X.index)
                    result = pd.concat([X, y_df], axis=1)
                    df_list.append(result)
            len_each_run.append(len(df_list))
                    
    # metdata_train_df = pd.concat(df_list[:-1], ignore_index=True) # all workers but one is the validation set
    # metdata_val_df = pd.DataFrame(df_list[-1]) # i want to take one seed from every run 
    # x_train = metdata_train_df.drop(columns=['reward_changes'])
    # y_train = metdata_train_df['reward_changes']
    # x_val = metdata_val_df.drop(columns=['reward_changes'])
    # y_val = metdata_val_df['reward_changes']
    

    # Subtract 1 from each index (to match Python indexing, which is 0-based)
    adjusted_indices = [i - 1 for i in len_each_run] # because i want to take 1 seed from every run!
    
    # Extract the selected DataFrames into one list
    metdata_val_df_list = [df_list[i] for i in adjusted_indices]
    # Extract the remaining DataFrames into another list
    metdata_train_df_list = [df for idx, df in enumerate(df_list) if idx not in adjusted_indices]
    # Combine the selected DataFrames into one DataFrame
    metdata_val_df = pd.concat(metdata_val_df_list, ignore_index=True)
    # Combine the remaining DataFrames into another DataFrame
    metdata_train_df = pd.concat(metdata_train_df_list, ignore_index=True)

    x_train = metdata_train_df.drop(columns=['reward_changes'])
    y_train = metdata_train_df['reward_changes']
    x_val = metdata_val_df.drop(columns=['reward_changes'])
    y_val = metdata_val_df['reward_changes']
    scaler = StandardScaler()
    #x_train['similiarity_feature'] =normalize(x_train['similiarity_feature'], [min(0, x_train['similiarity_feature'].min()), x_train['similiarity_feature'].max()]) # 0 because the test set will have a 0 sim to itself
    #x_val['similiarity_feature'] = normalize(x_val['similiarity_feature'], [min(0, x_val['similiarity_feature'].min()), x_val['similiarity_feature'].max()])
    #x_train[time_attr] = scaler.fit_transform(x_train[time_attr]) # 0 because the test set will have a 0 sim to itself
    #x_val[time_attr] = scaler.transform(x_val[time_attr])
    if partition_val:
        return x_train, y_train, x_val, y_val
    else:
        # Combine training and validation sets
        x_combined = pd.concat([x_train, x_val], ignore_index=True)
        y_combined = pd.concat([y_train, y_val], ignore_index=True)
        return x_combined, y_combined

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
if __name__ == "__main__":
    from DKL.pretraining_dkl import GPRegressionModel_DKL, LargeFeatureExtractor
    x_t, y_t, x_v, y_v = process_metadata(#['/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_20:56:29_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_length_0.05'],
                    [
                     '/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.30_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c16/CARLBipedalWalker_TERRAIN_LENGTH_320.0_pb2_Size_8_timesteps_total/seed9',
                     '/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.30_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c17/CARLBipedalWalker_TERRAIN_LENGTH_340.0_pb2_Size_8_timesteps_total/seed9',
                     '/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.30_pb2_job_scripts.CARLBipedalWalker.TERRAIN_LENGTH.c20/CARLBipedalWalker_TERRAIN_LENGTH_400.0_pb2_Size_8_timesteps_total/seed4'
                     ], #'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_20:56:29_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_length_0.05',
                    {
                    #"lambda": [0.9, 1.0],
                    #"clip_param": [0.1, 0.5],
                    #'gamma': [0.9,0.99],
                    "lr": [1e-5, 1e-3],
                    #"train_batch_size": [1000, 10_000],
                    'num_sgd_iter': [3,30]
                    #'entropy_coeff' : [0.01, 0.5]
                }
                ,current_env_config= {'env_name':'CARLBipedalWalker','TERRAIN_LENGTH' :320})
    # x_test, y_test = process_metadata(#['/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_20:56:29_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_length_0.05'],
    #                 ['/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_20:56:29_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_length_0.05'
    #                  #'/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/pb2.gravity.c2/2024-09-15_18:52:35_PPO_gravity_1.9600000000000002_pb2_Size8_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_gravity_1.9600000000000002'
    #                  ],
    #                 {
    #                 #"lambda": [0.9, 1.0],
    #                 #"clip_param": [0.1, 0.5],
    #                 #'gamma': [0.9,0.99],
    #                 "lr": [1e-5, 1e-3],
    #                 #"train_batch_size": [1000, 10_000],
    #                 'num_sgd_iter': [3,30]
    #                 #'entropy_coeff' : [0.01, 0.5]
    #             }
    #             ,current_env_config= {'env_name':'CARLCartPole','length' :0.5}, partition_val=False)
    

    neuralnet =  LargeFeatureExtractor(data_dim=5, seed=0)
    l1 = gpytorch.likelihoods.GaussianLikelihood()
    train_x = torch.tensor(x_t.values, dtype=torch.float32)
    train_y =torch.squeeze(torch.tensor(y_t.values, dtype=torch.float32))
    gp_model = GPRegressionModel_DKL(train_x,train_y,l1, 0, neuralnet,append_sim_column=False)
    test_x = torch.tensor(x_v.values, dtype=torch.float32)
    test_y =torch.squeeze(torch.tensor(y_v.values, dtype=torch.float32))

    metatrain_DKL_wilson(gp_model, train_x, train_y, l1, 0, save=True, X_test=test_x, y_test=test_y, training_iterations=100)
    







    
