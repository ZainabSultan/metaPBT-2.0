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
from DKL.pretraining_dkl import PB2_dkl_pretrained
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms import ppo
from copy import deepcopy
from CARL_env_reg_wrapper import CARLWrapper


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

def env_creator(env_config):
    print(env_config)
    env_name = env_config.pop('env_name')
    if env_name == 'CARLCartPole':
        from CARL_env_reg_wrapper_copy import CARLCartPoleWrapper as env
    elif env_name == 'CARLMountainCar':
        from CARL_env_reg_wrapper_copy import CARLMountainCarWrapper as env
    elif env_name == 'CARLMountainCarCont':
        from CARL_env_reg_wrapper_copy import CARLMountainCarContWrapper as env
    elif env_name == 'CARLAcrobot':
        from CARL_env_reg_wrapper_copy import CARLAcrobotWrapper as env
    elif env_name == 'CARLPendulum':
        from CARL_env_reg_wrapper_copy import CARLPendulumWrapper as env
    else:
        raise NotImplementedError
    return env(contexts={0:env_config})

def generate_sampler(seed, dist_type, bounds_tuple, num_samples):
            # loguniform_dist = tune.loguniform(1e-5, 1e-3)
        # samples_lr = [loguniform_dist.sample() for _ in range(args.num_samples)]
        
        # sample_iter_lr = iter(samples_lr)
        # get_lr_sample = lambda: next(sample_iter_lr)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        dist = dist_type(bounds_tuple[0], bounds_tuple[1])
        samples = [dist.sample() for _ in range(num_samples)]
        sample_iterator = iter(samples)
        get_sample_func = lambda: next(sample_iterator)
        return get_sample_func

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
    parser.add_argument('--metadata_dir_list', nargs='+', help='List of directories', required=True)

    args = parser.parse_args()
    context = json.loads(args.context.replace("'", ""))
    context_= context | {'env_name':args.env_name}  # includes env name for the env creator
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if True:
        register_env( args.env_name, env_creator)


        if args.env_name in ["BipedalWalker-v2", "BipedalWalker-v3"]:
            horizon = 1600
        else:
            horizon = 1000

        timelog = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


        args.dir = os.path.join(args.ws_dir , "{}_{}_{}_{}_Size{}_{}_{}".format(
            timelog,
            args.algo,
            dict_to_path_string(context),
            args.scheduler,
            str(args.num_samples),
            args.env_name,
            args.criteria,
        ))

        # meta_data_dir_list = ['/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/2024-09-12_19:58:08_PPO_length_0.05_pb2_Size4_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_length_0.05',
        #                       '/pfs/work7/workspace/scratch/fr_zs53-dkl_exps/cluster_logs/pb2.gravity.c1/2024-09-15_18:44:38_PPO_gravity_0.9800000000000001_pb2_Size8_CARLCartPole_timesteps_total/pb2_CARLCartPole_seed0_gravity_0.9800000000000001']

        meta_data_dir_list = args.metadata_dir_list
        print(meta_data_dir_list)
        pb2_metadkl=PB2_dkl_pretrained(
            time_attr=args.criteria,
            #metric=args.metric,
            #mode="max",
            perturbation_interval=args.t_ready,
            quantile_fraction=args.perturb,  # copy bottom % with top %
            # Specifies the hyperparam search space
            hyperparam_bounds={
                #"lambda": [0.9, 1.0],
                #"clip_param": [0.1, 0.5],
                "lr": [1e-5, 1e-3],
                #"train_batch_size": [1000, 5000],
                'num_sgd_iter': [3,30]
                #"train_batch_size": [1000, 60000],
            },
            seed=args.seed,
            synch=True,
            meta_data_dir_list=meta_data_dir_list,
            current_env_config=deepcopy(context_)
            #save_path=args.dir
        )
        


        schedulers = { 'metadkl':pb2_metadkl}
        # # LR SAMPLER
        # loguniform_dist = tune.loguniform(1e-5, 1e-3)
        # samples_lr = [loguniform_dist.sample() for _ in range(args.num_samples)]
        
        # sample_iter_lr = iter(samples_lr)
        # get_lr_sample = lambda: next(sample_iter_lr)


        config = PPOConfig()
        config.environment(env=args.env_name, env_config=context_)
        config.seed = args.seed
        config.rollouts(num_envs_per_worker=1)
        config.training(
            #lr= tune.sample_from(generate_sampler(args.seed, tune.loguniform, (1e-5, 1e-3), args.num_samples)), #tune.loguniform(1e-5, 1e-3),
            #kl_coeff = 1.0,
            #num_sgd_iter = tune.qrandint(3,30)#tune.sample_from(generate_sampler(args.seed,tune.qrandint, (3,30), args.num_samples))#tune.qrandint(3,30)
            #sgd_minibatch_size = tune.choice([128, 512, 2000]),
            #train_batch_size= tune.qrandint(1000,10_000),#tune.choice([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]),
            #gamma=tune.uniform(0.9, 0.99),
        #entropy_coeff=tune.uniform(0.01, 0.5),
            lr=tune.loguniform(1e-5, 1e-3),
            #kl_coeff = 1.0,
            num_sgd_iter = tune.qrandint(3,30)

        )
        run_name= "{}_{}_seed{}_{}".format(
                args.scheduler, args.env_name, str(args.seed), dict_to_path_string(context)
            )
        #config.debugging(log_level ="WARN")
        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                metric=args.metric,
                mode="max",
                scheduler=schedulers[args.scheduler],
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

        # Print `path` where checkpoints are stored
        print('Best result path:', best_result.path)

        # Print the best trial `config` reported at the last iteration
        # NOTE: This config is just what the trial ended up with at the last iteration.
        # See the next section for replaying the entire history of configs.
        print("Best final iteration hyperparameter config:\n", best_result.config)

        # Plot the learning curve for the best trial
        df = best_result.metrics_dataframe
        # Deduplicate, since PBT might introduce duplicate data
        df = df.drop_duplicates(subset=args.criteria, keep="last")
        df.plot(args.criteria, args.metric)
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(os.path.join(args.dir,run_name,'best_agent.png'))
    # except Exception as e:
    #     print(e)
