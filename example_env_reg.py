

import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from carl.envs.gymnasium.classic_control import CARLMountainCar
from ray.tune.registry import register_env
import numpy as np
import torch
import matplotlib.pyplot as plt
from CARL_env_reg_wrapper import CARLWrapper
from gymnasium import spaces

context = {'gravity': 0.0025}
wrapper_context = {0:context}
# ray.init()
# algo = ppo.PPO(env=CARLMountainCar, config={
#     "env_config": {"contexts": wrapper_context},  # config to pass to env class
# })

def env_creator(env_config):
    #base_env = CARLMountainCar(contexts=env_config)
    #print('base',base_env.observation_space)
    #base_env_space = spaces.Box(shape=(2,), low=-np.inf, high=np.inf)
    #env= CARL_env_wrapper(base_env, base_env_space)
    env = CARLWrapper(contexts={0:env_config})
    return env  # return an env instance


register_env("CARLMountainCar", env_creator)
#algo = ppo.PPO(env="CARLMountainCar")


import random

import ray
from ray import train, tune
from DKL.pb2 import PB2

#if __name__ == "__main__":
    #import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--smoke-test", action="store_true", help="Finish quickly for testing"
    # )
    # args, _ = parser.parse_known_args()

    # Postprocess the perturbed config to ensure it's still valid
    # def explore(config):
    #     # ensure we collect enough timesteps to do sgd
    #     if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
    #         config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    #     # ensure we run at least one sgd iter
    #     if config["num_sgd_iter"] < 1:
    #         config["num_sgd_iter"] = 1
    #     return config

hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    #"num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    #"train_batch_size": lambda: random.randint(2000, 160000),
}

pbt = PB2(
    time_attr="time_total_s",
    perturbation_interval=5,
    #resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_bounds={"lr": [0.0001, 0.1]},
    #custom_explore_fn=explore,
    seed=0,
    synch=True,
)

# Stop when we've either reached 100 training iterations or reward=300
stopping_criteria = {"training_iteration": 20, "episode_reward_mean": 900}
pertubation_interval=5
tuner = tune.Tuner(

    "PPO",
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        scheduler=pbt,
        num_samples=2,
        
    ),
    param_space={
        "env": "CARLMountainCar",
        "kl_coeff": 1.0,
        "num_workers": 2,
        "num_cpus": 1,  # number of CPUs to use per trial
        "num_gpus": 0,  # number of GPUs to use per trial
        #"model": {"free_log_std": True},
        "env_config": context,
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
         "checkpoint_interval": pertubation_interval
        # These params start off randomly drawn from a set.
        #"num_sgd_iter": tune.choice([10, 20, 30]),
        #"sgd_minibatch_size": tune.choice([128, 512, 2048]),
        #"train_batch_size": tune.choice([10000, 20000, 40000]),
    },
    run_config=train.RunConfig(stop=stopping_criteria,  storage_path="/Users/zasulta/Documents/DL/23_05_2024_DL_test/DKL/visualisations/ray_results",
                               name='CARLMCD'),
)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
results_grid = tuner.fit()
best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")

# Print `path` where checkpoints are stored
print('Best result path:', best_result.path)

# Print the best trial `config` reported at the last iteration
# NOTE: This config is just what the trial ended up with at the last iteration.
# See the next section for replaying the entire history of configs.
print("Best final iteration hyperparameter config:\n", best_result.config)

# Plot the learning curve for the best trial
df = best_result.metrics_dataframe
# Deduplicate, since PBT might introduce duplicate data
df = df.drop_duplicates(subset="training_iteration", keep="last")
df.plot("training_iteration", "mean_accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Test Accuracy")
plt.show()