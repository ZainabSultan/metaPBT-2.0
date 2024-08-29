## generate trajs
## 
from datetime import datetime
import json
import os
from carl.envs.gymnasium.classic_control import CARLMountainCar
from stable_baselines3 import PPO
import random

from CARL_env_reg_wrapper import CARLWrapper


CONTEXT_VAR = 'gravity'
policies_save_dir = 'PPO'
trajectories_save_dir = 'trajectories'
SEED=0
NUM_STATES=5
random_config = {'num_states_to_select': NUM_STATES, 'seed':SEED}


def generate_trajectories(env, pi_star_file_path):
    # for each env with pi_star
    context = env.env.contexts[0][CONTEXT_VAR]
    history= {
        'context': context,
        'policy': pi_star_file_path,
        'state_trajectory': [],
        'action_trajectory':[]
    }
    pi_star = PPO.load(pi_star_file_path)
    obs, _ = env.reset()
    print(obs, 'obs from reset')
    i=0
    terminate, truncate = False, False
    while not terminate and not truncate: # or truncate
        history['state_trajectory'].append(obs.tolist())
        action, _ = pi_star.predict(obs)
        history['action_trajectory'].append(str(action))
        obs, rewards, terminate, truncate, info = env.step(action)
        print(terminate, truncate, i)
        i=i+1
    return history
    # save




def save(history, trajectories_file_path):
    # date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # env_name = env.env.env_name
    # trajectories_dir_path = os.path.join(trajectories_save_dir, env_name)
    if not os.path.exists(trajectories_file_path):
        os.makedirs(trajectories_file_path)
    # trajectories_file_name = '{date}.json'.format(date = date)
    # trajectories_file_path = os.path.join(trajectories_dir_path, trajectories_file_name)
    with open(trajectories_file_path, 'w+') as f:
        json.dump(history,f )


def vine_prev(env_target: CARLWrapper, trajectory_file_path_source,trajectory_file_path_target, subset_selection_strategy: str, pi_star_file_path):
    # load file
    with open(trajectory_file_path_source, 'rb') as f:
        content_source = json.load(f)
    with open(trajectory_file_path_target, 'rb') as f:
        content_target = json.load(f)

    random.seed(random_config['seed'])
    policy = PPO.load(content_source['policy']) # or target, both should be same
    
    if subset_selection_strategy == 'RANDOM':
        states = content_source['state_trajectory']
        selected_states = random.sample(states, random_config['num_states_to_select'])
    
    if selected_states is not None:
        # vine method to force env to be in these states
        for s in selected_states:
            env_target.reset()
            env_target.state = s
            trajectory = generate_trajectories(env_target, policy)
            content_target['state_trajectory'].extend(trajectory['state_trajectory'])
            content_target['action_trajectory'].extend(trajectory['action_trajectory'])


def vine(env, state_list, trajectory_file_path):
    
        
            



def create_env(env_family='carl_mcd', env_context={0:{'gravity': 0.0025}}):
    if env_family == 'carl_mcd':
        env= CARLWrapper(contexts=env_context)
    env.reset(SEED)
    return env
 

def EPI_data_gen(env_family_name, context_features_list):

    # train a policy first ? on the defualt
    policy_pi = ...

    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_dir = env_family_name + '_' + date
    # first generate trajectories for all envs
    collective_states = []
    for context_feature in context_features_list:
        env = create_env(env_family_name, context_feature)
        history = generate_trajectories(env, '')
        collective_states.extend(history['state_trajectory'])
        ctxt_ft_suffix = str(context_feature[0])
        save(history, save_dir + '{ctxt_ft_suffix}.json'.format(ctxt_ft_suffix=ctxt_ft_suffix))

    # pick at random some states and force envs to be at it ??
    random.seed(random_config['seed'])
    selected_states = random.sample(collective_states, random_config['num_states_to_select'])



        

def main():
    context_features_list = [{0:{'gravity': 0.0025}}, {0:{'gravity': 0.0015}}]




    





    