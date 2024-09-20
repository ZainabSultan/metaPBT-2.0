from DKL.pretraining_dkl_utils import process_metadata, pretrain_neural_network_model_with_sched
from DKL.pretraining_dkl


if __name__ == "__main__":
    paths_list= []
    x_t, y_t, x_v, y_v = process_metadata(meta_data_dir_list=paths_list,
                   hyperparams_bounds= {
                    "lr": [1e-5, 1e-3],
                    'num_sgd_iter': [3,30]
                }
                ,current_env_config= {'env_name':'CARLCartPole','length' :0.5}) 

    pretrain_neural_network_model_with_sched()