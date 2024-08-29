import gymnasium as gym
from typing import Optional
import numpy as np



class CARL_env_wrapper(gym.ObservationWrapper):
    def __init__(self, env, space, hide_context=True):
        super().__init__(env)
        self.hide_context=hide_context
        #self.observation_space = space

    @staticmethod
    def typecast_obs(obs):
        obs_array = np.concatenate([value.flatten() for value in obs.values()])
        return obs_array
        
        
#FOR Debugg purposes
    def observation(self, obs):
        return obs
    
    def reset(self,**kwargs):
        if self.hide_context:
            state, info= self.env.reset(**kwargs)
            return self.typecast_obs(state['obs']), info
        else:
            print('idk')
            raise Exception
    
    def step(self, action):
        
        obs, reward, terminate, truncate, info= self.env.step(action)
        obs = self.typecast_obs(obs['obs'])
        return obs,reward, terminate, truncate, info

        
