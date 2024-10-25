


from typing import Type, Dict, Any, Union, Optional
from gym import Env  # assuming gym.Env is the base class for environments
from carl.envs.gymnasium.classic_control import CARLMountainCar, CARLCartPole, CARLAcrobot, CARLMountainCarContinuous, CARLPendulum
from typing import Any, Dict
from carl.context.selection import AbstractSelector
from gymnasium import Env
from carl.envs.gymnasium.box2d import CARLBipedalWalker, CARLLunarLander
import numpy as np
import torch
from DKL.EPI.models import evaluate_model


class CARLWrapperBaseEPI(Env):
    def __init__(
        self,
        env_class: Type[Env],
        env: Optional[Env] = None,
        contexts: Optional[Dict[Any, Dict[str, Any]]] = None,
        obs_context_features: Optional[list[str]] = None,
        obs_context_as_dict: bool = True,
        context_selector: Optional[Union[AbstractSelector, Type[AbstractSelector]]] = None,
        context_selector_kwargs: Optional[dict] = None,
        vanilla_model= None,
        epi_embedding_conditioned_model = None,
        val_data = None,
        **kwargs
    ) -> None:

        self.env = env_class(env, contexts, obs_context_features, obs_context_as_dict, context_selector, context_selector_kwargs, **kwargs)
        self.observation_space = self.env.base_observation_space
        self.contexts = contexts
        self.action_space = self.env.action_space
        self.state_trajectory = np.array([])
        self.action_trajectory = np.array([])
        self.goal_traj_length = 10
        self.vanilla_model = vanilla_model
        self.epi_embedding_conditioned_model = epi_embedding_conditioned_model
        self.val_data=val_data
        
    def step(self, action):
        obs_context, r, term, trun, info = self.env.step(action)
        obs = obs_context['obs']
        current_context = obs_context['context']
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        self.state = obs
        if action.shape == ():
                action = np.array([action])
        
        self.action_trajectory = np.concatenate((self.action_trajectory, action), axis=0)
        self.state_trajectory = np.concatenate((self.state_trajectory, obs), axis=0)
        r = 0

        if len(self.action_trajectory) == self.goal_traj_length:
            #print(current_context, ' should be switched the next time')
            #print(self.state_trajectory, self.action_trajectory)
            epi_trajectory = np.concatenate((self.state_trajectory, self.action_trajectory), axis = 0)
            #print(epi_trajectory, ' _this is the trajectory')
            r = self.calculate_reward(epi_trajectory, current_context) #ä TODO HOW TO ACCESS CURRENT CONTEXT
            self.state_trajectory = np.array([])
            self.action_trajectory = np.array([])
            term = True
        info['context'] = current_context
        return obs, r, term, trun, info
    
    def reset(self, seed=None, options=None):
        obs_context, info = self.env.reset(seed=seed, options=options)
        obs = obs_context['obs']
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        obs_context['obs'] = obs
        self.state_trajectory = np.array(obs)
        self.action_trajectory = np.array([])
        info['context'] = obs_context['context']
        return obs, info
    
    def calculate_reward(self, epi_trajectory, current_context):
        epi_trajectory = torch.tensor(epi_trajectory, dtype=torch.float32)
        #print('loss epi')
        loss_epi = evaluate_model(model=self.epi_embedding_conditioned_model, embedding=epi_trajectory, env_identifier=current_context, val_df=self.val_data)
        #print(' loss vanilla')
        loss_vanilla = evaluate_model(model = self.vanilla_model,env_identifier=current_context, val_df=self.val_data )
        reward = loss_vanilla - loss_epi
        return reward        

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    

# Example Wrappers for Specific Environments
class CARLLunarLanderWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLLunarLander, *args, **kwargs)

class CARLBipedalWalkerWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLBipedalWalker, *args, **kwargs)

class CARLMountainCarWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLMountainCar, *args, **kwargs)

class CARLAcrobotWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLAcrobot, *args, **kwargs)

class CARLCartPoleWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLCartPole, *args, **kwargs)

class CARLMountainCarContWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLMountainCarContinuous, *args, **kwargs)

class CARLPendulumWrapper(CARLWrapperBaseEPI):
    def __init__(self, *args, **kwargs):
        super().__init__(CARLPendulum, *args, **kwargs)

def generate_trajs(model, env, len_trajectory=10):
    obs, info = env.reset()
    trajectories_batch ={}
    n_envs = len(env.contexts)
    for _ in range(n_envs):
        traj = obs
        for _ in range(len_trajectory):
            action, _ = model.predict(obs, deterministic=True)
            # Take a step in the environment
            next_obs,  r, term, trun, info= env.step(action)
            if action.shape == ():
                action = np.array([action])
            traj = np.concatenate((traj, action, next_obs), axis=0)
        trajectories_batch[str(info['context'])] = traj
        obs, info = env.reset() # force change of env
    return trajectories_batch
        


    
    


