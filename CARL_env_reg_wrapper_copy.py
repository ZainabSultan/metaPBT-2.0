from typing import Any, Dict, Type
from gym import Env  # assuming gym.Env is the base class for environments
from carl.envs.gymnasium.classic_control import CARLMountainCar, CARLCartPole, CARLAcrobot, CARLMountainCarContinuous, CARLPendulum
from typing import Any, Dict
from carl.context.selection import AbstractSelector
from gymnasium import Env

class CARLWrapperBase(Env):
    def __init__(
        self,
        env_class: Type[Env],
        env: Env | None = None,
        contexts: Dict[Any, Dict[str, Any]] | None = None,
        obs_context_features: list[str] | None = None,
        obs_context_as_dict: bool = True,
        context_selector: AbstractSelector | type[AbstractSelector] | None = None,
        context_selector_kwargs: dict = None,
        **kwargs
    ) -> None:
        self.env = env_class(env, contexts, obs_context_features, obs_context_as_dict, context_selector, context_selector_kwargs, **kwargs)
        self.observation_space = self.env.base_observation_space
        self.action_space = self.env.action_space
        

    def step(self, action):
        obs_context, r, term, trun, info = self.env.step(action)
        obs = obs_context['obs']
        self.state = obs
        return obs, r, term, trun, info

    def reset(self, seed=None, options=None):
        obs_context, info = self.env.reset(seed=seed, options=options)
        obs = obs_context['obs']
        self.state = obs
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# Example Wrappers for Specific Environments

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


env_config =  {'env_config': {'gravity': 0.01}, 'env_name': 'CARLMountainCar'}
# env = CARLMountainCarWrapper(contexts={0:env_config})
# env.reset(seed=0)
# print(env.step(0))

# env_config =  {'LINK_LENGTH_1': 1}
# env = CARLAcrobotWrapper(contexts={0:env_config})
# env.reset(seed=0)
# print(env.step(0))
from ray.rllib.algorithms import ppo

from ray.tune.registry import register_env
def env_creator(env_config):
    env_name = env_config['env_name']
    env_context = env_config['env_config']
    if env_name == 'CARLCartPole':
        env = CARLCartPoleWrapper(contexts={0:env_context})
    elif env_name == 'CARLMountainCar':
        env = CARLMountainCarWrapper(contexts={0:env_context})
    elif env_name == 'CARLMountainCarCont':
        env=CARLMountainCarContWrapper(contexts={0:env_context})
    elif env_name == 'CARLAcrobot':
        env=CARLAcrobotWrapper(contexts={0:env_context})
    elif env_name == 'CARLPendulum':
        env=CARLPendulumWrapper(contexts={0:env_context})
    else:
        raise NotImplementedError

    return env

# env_name='CARLMountainCar'

# register_env(env_name, env_creator)
# algo = ppo.PPO(env="CARLMountainCar",config={
#     "env_config": env_config,  # config to pass to env class
# })
