from typing import Type, Dict, Any, Union, Optional
from gym import Env  # assuming gym.Env is the base class for environments
from carl.envs.gymnasium.classic_control import CARLMountainCar, CARLCartPole, CARLAcrobot, CARLMountainCarContinuous, CARLPendulum
from typing import Any, Dict
from carl.context.selection import AbstractSelector
from gymnasium import Env
from carl.envs.gymnasium.box2d import CARLBipedalWalker, CARLLunarLander
import numpy as np



class CARLWrapperBase(Env):
    def __init__(
        self,
        env_class: Type[Env],
        env: Optional[Env] = None,
        contexts: Optional[Dict[Any, Dict[str, Any]]] = None,
        obs_context_features: Optional[list[str]] = None,
        obs_context_as_dict: bool = True,
        context_selector: Optional[Union[AbstractSelector, Type[AbstractSelector]]] = None,
        context_selector_kwargs: Optional[dict] = None,
        seed=None,
        **kwargs
    ) -> None:
        self.env = env_class(env, contexts, obs_context_features, obs_context_as_dict, context_selector, context_selector_kwargs, **kwargs)
        self.observation_space = self.env.base_observation_space
        self.action_space = self.env.action_space
        self.seed = seed

    def step(self, action):
        
        obs_context, r, term, trun, info = self.env.step(action)
        obs = obs_context['obs']
        #assert not np.isnan(obs).any(), "Observation contains NaN values!"
        #assert not np.isnan(action).any(), "ACTIONS contains NaN values!"
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        self.state = obs
        return obs, r, term, trun, info
    
    def reset(self, seed=None, options=None):
        if seed:
            self.seed=seed
        obs_context, info = self.env.reset(seed=self.seed, options=options)
        obs = obs_context['obs']
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        self.state = obs
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# Example Wrappers for Specific Environments
class CARLLunarLanderWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLLunarLander, *args, **kwargs)


class CARLBipedalWalkerWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLBipedalWalker, *args, **kwargs)

class CARLMountainCarWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLMountainCar, *args, **kwargs)

class CARLAcrobotWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLAcrobot, *args, **kwargs)

class CARLCartPoleWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLCartPole, *args, **kwargs)

class CARLMountainCarContWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLMountainCarContinuous, *args, **kwargs)

class CARLPendulumWrapper(CARLWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLPendulum, *args, **kwargs)
    

def env_creator(env_config):
    print(env_config)
    env_name = env_config.pop('env_name')
    if 'seed' not in env_config:
        seed = 0
    else:
        seed = env_config.pop('seed')
    if env_name == 'CARLCartPole':
        env = CARLCartPoleWrapper
    elif env_name == 'CARLMountainCar':
        env = CARLMountainCarWrapper
    elif env_name == 'CARLMountainCarCont':
        env = CARLMountainCarContWrapper
    elif env_name == 'CARLAcrobot':
        env = CARLAcrobotWrapper
    elif env_name == 'CARLPendulum':
        env = CARLPendulumWrapper
    elif env_name == 'CARLLunarLander':
        env = CARLLunarLanderWrapper
    elif env_name == 'CARLBipedalWalker':
        env = CARLBipedalWalkerWrapper
    else:
        raise NotImplementedError
    return env(contexts={0:env_config}, seed=seed)

