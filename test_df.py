


from DKL.Meta_DKL_data_utils import process_pb2_runs_metadata


path_in_q ='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/2024-09-22_18:04.31_pb2_job_scripts.CARLBipedalWalker.TERRAIN_STEP.c10/CARLBipedalWalker_TERRAIN_STEP_0.4666666666666667_pb2_Size_8_timesteps_total/seed9'

hp_bounds = {
                "lambda": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                #'gamma': [0.9,0.99],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 10_000],
                'num_sgd_iter': [3,30]
             }

curr_env = {'TERRAIN_STEP': 0.4666666666666667}

X, y = process_pb2_runs_metadata([path_in_q], current_env_config=curr_env, hyperparams_bounds=hp_bounds, partition_val=False)

print(X.describe())
print(y.describe())