

from CARLWrapper import env_creator
import numpy as np

env_config = {'env_name':'CARLBipedalWalker','TERRAIN_LENGTH': 0.05, 'seed':0}

env = env_creator(env_config)

state_ = env.reset()
state_ac = env.step([0.0,0.0,0.0,0.0])
state_ac = env.step([0.2,0.0,0.0,0.0])
env_config = {'env_name':'CARLBipedalWalker','TERRAIN_LENGTH': 0.05, 'seed':0}

env = env_creator(env_config)

state = env.reset()
state_ac2 = env.step([0.0,0.0,0.0,0.0])
state_ac2 = env.step([0.2,0.0,0.0,0.0])
print(state)
print(state_)
print('next states: ')
print(state_ac)
print(state_ac2)
if np.array_equal(state[0], state_[0]):
    print('same start')
if np.array_equal(state_ac[0], state_ac2[0]):
    print('same next states')