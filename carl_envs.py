
from CARL import carl
from carl.envs.gymnasium.classic_control import CARLMountainCar
context = {'gravity': 0.0025}
wrapper_context = {0:context}
base_env = CARLMountainCar(contexts=wrapper_context)
print('base',base_env.contexts)

import gym
from stable_baselines3 import PPO
from ray import tune

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