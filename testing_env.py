
from carl.envs.gymnasium.classic_control import CARLMountainCar
env_config =  {'gravity': 0.0025}
env = CARLMountainCar(contexts={0:env_config})
print( env._observation_space)
env.reset()
#print(env.reset()[0]['obs'])
print(env.step(0)[0]['obs'])

# from ray.rllib.algorithms import ppo

# from ray.tune.registry import register_env

# def env_creator(env_config):
#     return CARLMountainCar(contexts={0:env_config})  # return an env instance

# register_env("my_env", env_creator)
# algo = ppo.PPO(env="my_env",config={
#     "env_config": env_config,  # config to pass to env class
# })
# print(algo.train())