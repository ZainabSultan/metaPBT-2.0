


from typing import Any, Dict
from carl.context.selection import AbstractSelector
from carl.envs.gymnasium.classic_control import CARLMountainCar
#from carl.utils.types import Contexts
from gymnasium import Env
#from CARL.carl.envs.gymnasium.classic_control import CARLMountainCar
#import gymnasium

class CARLWrapper(CARLMountainCar):
    def __init__(self, env: Env | None = None, contexts: Dict[Any, Dict[str, Any]] | None = None, obs_context_features: list[str] | None = None, obs_context_as_dict: bool = True, context_selector: AbstractSelector | type[AbstractSelector] | None = None, context_selector_kwargs: dict = None, **kwargs) -> None:
        super().__init__(env, contexts, obs_context_features, obs_context_as_dict, context_selector, context_selector_kwargs, **kwargs)
        self.observation_space = self.base_observation_space
        

    def step(self,action):
        obs_context, r, term, trun, info = super().step(action)
        obs = obs_context['obs']
        self.state = obs
        return obs, r, term, trun, info
    
    def reset(self, seed, options=None):
        obs_context, info = super().reset(seed=seed, options=options)
        obs = obs_context['obs']
        self.state=obs
        return obs, info
    



# env_config =  {'gravity': 0.0025}
# env = CARLWrapper(contexts={0:env_config})
# env.reset(seed=0)
# print(env.step(0))

#env_config =  {'gravity': 0.1}
# env = CARLWrapper(contexts={0:env_config})
# env.reset(seed=0)
# print(env.step(0))


# env_config_2 =  {'gravity': 0.1}
# env = CARLWrapper(contexts={0:env_config_2})
# env.reset(seed=0)
# print(env.step(0))


# from ray.rllib.algorithms import ppo

# from ray.tune.registry import register_env

# def env_creator(env_config):
#     print(env_config)
#     env=CARLWrapper(contexts={0:env_config})
#     print(env.contexts)
#     return env   # return an env instance

# register_env("mcd", env_creator)

# env1 = gymnasium.make("mcd", env_config=env_config)
# env1.reset(seed=0)
# print(env1.step(0))
# # Create the second environment instance
# env2 = gymnasium.make("mcd", env_config=env_config_2)
# env2.reset(seed=0)
# print(env2.step(0))

# algo = ppo.PPO(env="mcd",config={
#     "env_config": env_config,  # config to pass to env class
# })
# print(algo.train())
# import gymnasium as gym
# from carl.envs.gymnasium.classic_control import CARLMountainCar
# import numpy as np


# class CARLMCD(CARLMountainCar):
#     def __init__(self, contexts:dict=None):
#         self.env_name="MountainCar-v0"
#         self.context = contexts
#         self.render_mode='rgb_array'
#         if 'hide_context' not in self.contexts.keys():
#             self.contexts['hide_context'] = True # default is hiding conetxt
#         env= gym.make(id=self.env_name, render_mode=self.render_mode)
#         super().__init__(contexts=contexts)
        
        
        
#     def reset(self, seed, options):
#         state, info = self.reset(seed=seed, options=options)
#         # # convert state and take out contexts if needed
#         if self.context['hide_context'] is True:
#             state.pop('contexts')
#         state_typecast= self.typecast_obs(state)
#         return state_typecast, info
        
#     def step(self, action):
#         obs, reward, term, trun, info = self.step(action =  action)
#         obs_typecast = self.typecast_obs(obs)
#         return obs_typecast, reward, term, trun, info

#     @staticmethod
#     def typecast_obs(obs):
#         obs_array = np.concatenate([value.flatten() for value in obs.values()])
#         return obs_array
    
# c = CARLMCD(contexts={0:{'gravity': 0.0025}})
