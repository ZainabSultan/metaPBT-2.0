
# from CARL import carl
# from carl.envs.gymnasium.classic_control import CARLMountainCar
# context = {'gravity': 0.0025}
# wrapper_context = {0:context}
# base_env = CARLMountainCar(contexts=wrapper_context)
# print('base',base_env.contexts)
import random

import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining

import argparse
import json
import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ray.tune.registry import register_env
import torch
from DKL.PB2_DKL import PB2_dkl
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.rllib.algorithms import ppo

from CARL_env_reg_wrapper import CARLWrapper
import gym
from stable_baselines3 import PPO
from ray import tune
print('here')
def train_ppo(config):
    env = gym.make(config["env_name"])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gae_lambda=config["gae_lambda"],
        gamma=config["gamma"],
        verbose=1,
    )
    model.learn(total_timesteps=config["total_timesteps"])
    tune.report(mean_reward=env.get_attr('reward'))

print('here')
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


print('here')
env_name='CARLCartPole'
context = {'gravity':11.0, 'env_name': 'CARLCartPole'}

env = env_creator(context)
print(env.env.contexts)
register_env(env_name, env_creator)
print('registered')
# algo = ppo.PPO(env=env_name, config={
#     "env_config": context,  # config to pass to env class
# })
# print(algo.__dict__)
# algo = ppo.PPO(env=env,config={
#     "env_config": {contexts:{0:context}},  # config to pass to env class
# })

# print(algo.env.contexts)