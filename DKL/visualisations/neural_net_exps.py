from DKL.pretraining_dkl_utils import process_metadata, pretrain_neural_network_model_with_sched, pretrain_neural_network_model_with_sched_reset
from DKL.pretraining_dkl import LargeFeatureExtractor
import numpy as np 

def baseline(y_t, y_val):
    mean_prediction = y_t.mean()  # Calculate the mean of the training set
    loss = np.square(y_val - mean_prediction)  # Squared difference from the validation set
    mse = np.mean(loss)  # Mean of the squared errors
    print(f'MSE: {mse}')
    return mse

if __name__ == "__main__":
    paths_list= ['/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/pb2.gravity.c10/2024-09-16_08:09:03_PPO_gravity_0.0025_pb2_Size8_CARLMountainCar_timesteps_total/pb2_CARLMountainCar_seed0_gravity_0.0025']#
    paths_list=['/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/pb2.gravity.c1/2024-09-15_18:44:38_PPO_gravity_0.9800000000000001_pb2_Size8_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_gravity_0.9800000000000001']
    x_t, y_t, x_v, y_v = process_metadata(meta_data_dir_list=paths_list,
                   hyperparams_bounds= {
                    "lr": [1e-5, 1e-3],
                    'num_sgd_iter': [3,30]
                }
                ,current_env_config= {'env_name':'CARLCartPole','length' :0.5}) 
    baseline(y_t, y_v)
        
    # model = LargeFeatureExtractor(data_dim=x_t.shape[1], seed=0)
    # pretrain_neural_network_model_with_sched_reset(model, x_t, y_t, x_v, y_v, seed=0)
    # print(y_t.describe())
    # print('val')
    # print(y_v.describe())