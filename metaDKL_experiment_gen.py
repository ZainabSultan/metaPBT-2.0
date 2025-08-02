import random

import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining

import argparse
import json
import os
import random
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ray.tune.registry import register_env
import torch
from DKL.copy_Meta_DKL import Meta_DKL
from CARLWrapper import env_creator
from copy import deepcopy
from context_mapper import get_context, get_context_id
def dict_to_path_string(params):
    # Flatten the dictionary into key=value format
    elements = [f"{key}_{value}" for key, value in params.items()]
    
    # Join the elements with underscores or any other separator
    path_friendly_string = "_".join(elements)
    
    # Replace any problematic characters if needed
    path_friendly_string = path_friendly_string.replace(" ", "_").replace("{", "").replace("}", "").replace(":", "")
    
    return path_friendly_string

# Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=30_000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--t_ready", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=str, default='{"gravity":0.00025}')
    parser.add_argument("--horizon", type=int, default=1600)
    parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
    parser.add_argument("--env_name", type=str, default="CARLMountainCar") #"CartPole-v1"
    parser.add_argument( "--criteria", type=str, default="timesteps_total")  
    parser.add_argument("--net", type=str, default="32_32") 
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument('--ws_dir', type=str, default=r'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir')
    parser.add_argument('--metric', type=str, default='env_runners/episode_reward_mean')
    parser.add_argument('--num_meta_envs', help='how many envs to use in the meta dataset', default=5)
    parser.add_argument('--meta_selection_method', help='how to select the envs, use gen for selecting general, equally spaced enviroments. Use def for selecting environments that are close to the default enviroment', default='gen')
    parser.add_argument('--meta_data_base_dir', default='/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/CARLMountainCar_4_agents/gravity', help='base directory for the meta data, where the meta data is stored')


    args = parser.parse_args()
    context = json.loads(args.context.replace("'", ""))
    
    context_= context | {'env_name':args.env_name}  # includes env name for the env creator

    id = get_context_id(args.env_name, context)
    id_list = np.arange(1,22)
    
    sorted_ids = sorted(id_list, key=lambda id_val: abs(id_val - id))
    if args.meta_selection_method == 'closest':
        closest_ids = sorted_ids[:int(args.num_meta_envs)]

        intersection = set(closest_ids).intersection([4,5,6,7])
        if intersection:
            print(f"Intersection found: {intersection}")

        else:
            print("No intersection found")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if True:
        register_env( args.env_name, env_creator)

        timelog = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            
        args.dir = os.path.join(args.ws_dir , "{}_{}_{}_Size_{}_{}".format(
                args.env_name,
                dict_to_path_string(context),
                f'MetaDKL_{args.meta_selection_method}_{args.num_meta_envs}',
                str(args.num_samples),
                args.criteria,
            ))

        current_context = get_context_id(args.env_name, context)
        
        pb2_metadkl=Meta_DKL(
            
            time_attr=args.criteria,
            perturbation_interval=args.t_ready,
            hyperparam_bounds={
                "lambda_": [0.9, 0.99],
                "clip_param": [0.1, 0.5],
                "lr": [1e-5, 1e-3],
                'num_sgd_iter': [3,30]
            },
            seed=args.seed,
            synch=True,
            meta_selection_method=args.meta_selection_method,
            num_meta_envs=int(args.num_meta_envs),
            current_env_config=deepcopy(context_),
            meta_data_base_dir=args.meta_data_base_dir
        )
        

        easy_envs = ['CARLMountainCar', 'CARLCartPole', 'CARLAcrobot', 'CARLPendulum', 'CARLMountainCarCont']
        config = PPOConfig()
        config.environment(env=args.env_name, env_config=context_)
        config.seed = args.seed
        config.rollouts(num_envs_per_worker=1)

        if args.env_name in easy_envs:
            config.training(
            lr=tune.loguniform(1e-5, 1e-3),
            grad_clip=2.5,
            clip_param = tune.uniform(0.1, 0.5),
            lambda_ = tune.uniform(0.9, 0.99),
            num_sgd_iter = tune.qrandint(3,30),
            train_batch_size= 1000

        )
            print('easy env activated')
        else:
            config.training(
            lr=tune.loguniform(1e-5, 1e-3),
            grad_clip=2.5,
            clip_param = tune.uniform(0.1, 0.5),
            lambda_ = tune.uniform(0.9, 0.99),
            num_sgd_iter = tune.qrandint(3,30)

        )
        run_name= "seed{}".format(
                 str(args.seed)
            )
        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                metric=args.metric,
                mode="max",
                scheduler=pb2_metadkl,
                num_samples=args.num_samples,
            
            ),
            param_space=config,
            run_config=train.RunConfig(name=run_name
                                       ,stop={args.criteria: args.max}, storage_path=args.dir,
                                       failure_config=train.FailureConfig(
            fail_fast=True  
        )),
                                       
            
        )
        results_grid = tuner.fit()
        best_result = results_grid.get_best_result(metric=args.metric, mode="max")
        print('Best result path:', best_result.path)
        print("Best final iteration hyperparameter config:\n", best_result.config)
        df = best_result.metrics_dataframe
        df = df.drop_duplicates(subset=args.criteria, keep="last")
        df.plot(args.criteria, args.metric)
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(os.path.join(args.dir,run_name,'best_agent.png'))

