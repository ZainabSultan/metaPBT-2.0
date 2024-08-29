import warnings
warnings.filterwarnings("ignore")
from CARL import carl
from carl.envs.gymnasium.classic_control import CARLMountainCar
from stable_baselines3 import PPO, DDPG
import os
import numpy as np 
from CARL_env_wrapper import CARL_env_wrapper
from gymnasium import spaces
from datetime import datetime
from gymnasium.wrappers.filter_observation import FilterObservation
import json
import itertools
from stable_baselines3.common.policies import obs_as_tensor


def generate_trajectories(env, pi_star_file_path):
    # for each env with pi_star
    context = env.env.contexts[0][CONTEXT_VAR]
    history= {
        'context': context,
        'policy': pi_star_file_path,
        'state_trajectory': [],
    }
    pi_star = PPO.load(pi_star_file_path)
    obs, _ = env.reset()
    print(obs, 'obs from reset')
    i=0
    terminate, truncate = False, False
    while not terminate and not truncate: # or truncate
        history['state_trajectory'].append(obs.tolist())
        action, _states = pi_star.predict(obs)
        obs, rewards, terminate, truncate, info = env.step(action)
        print(terminate, truncate, i)
        i=i+1
    # save
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    env_name = env.env.env_name
    trajectories_dir_path = os.path.join(trajectories_save_dir, env_name)
    if not os.path.exists(trajectories_dir_path):
        os.makedirs(trajectories_dir_path)
    trajectories_file_name = '{date}.json'.format(date = date)
    trajectories_file_path = os.path.join(trajectories_dir_path, trajectories_file_name)
    with open(trajectories_file_path, 'w+') as f:
        json.dump(history,f )
    
    


def train_policy(env, steps=25000):
    print(env.env.contexts)
    context = env.env.contexts[0][CONTEXT_VAR]
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    env_name = env.env.env_name
    save_subdir = env_name
    model_save_dir = os.path.join(policies_save_dir, save_subdir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    policy_save_path = os.path.join(model_save_dir,"{variable}_{context}_{date}".format(variable = CONTEXT_VAR, 
                                                                                        context = context, date=date))
    model.save(policy_save_path)
    return policy_save_path


def l1_distance(p1, p2):
    action_1, action_2 = np.argmax(p1), np.argmax(p2)
    return np.linalg.norm((action_1 - action_2), ord=1)


def tv_distance(p1, p2):
    return 0.5 * np.sum(np.abs(p1 - p2))

def predict_proba(policy, state):
    
    state = np.array(state)
    obs = policy.policy.obs_to_tensor(state)[0]
    dis = policy.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.detach().numpy()
    return probs_np

def DIST(trajectories_pair, pi_star_file_path_pair, has_discrete_actions, tolerance = 1e-7, gamma=0.99):
    
    states_env_1, states_env_2 = trajectories_pair[0], trajectories_pair[1]
    pi_star_env_1, pi_star_env_2 = PPO.load(pi_star_file_path_pair[0]), PPO.load(pi_star_file_path_pair[1])
    cost_matrix = np.zeros((len(states_env_1), len(states_env_2)))
    cost_matrix_current = np.zeros((len(states_env_1), len(states_env_2)))
    while True:
        print('another iteration')
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[0])):
                if has_discrete_actions:
                    dist = tv_distance
                else:
                    dist = l1_distance
                env_state_1, env_state_2 = states_env_1[i], states_env_2[j]
                #action_dist_1, _ = pi_star_env_1.predict(env_state_1, deterministic=False)
                #action_dist_2, _ = pi_star_env_2.predict(env_state_2, deterministic=False)
                action_dist_1 = predict_proba(pi_star_env_1, env_state_1)
                action_dist_2 = predict_proba(pi_star_env_2, env_state_2)
                cost_matrix_current[i,j] = dist(action_dist_1, action_dist_2) + (gamma *cost_matrix[i,j])
                if i == (len(cost_matrix) -1) and j == (len(cost_matrix[0]) -1): 
                    # distance between terminal states is 0
                    cost_matrix_current[i,j] = 0
        #absolute_difference = np.abs(cost_matrix_current - cost_matrix)
        #sum_of_differences = np.sum(absolute_difference)
        #print(sum_of_differences)
        if np.allclose(cost_matrix_current, cost_matrix, atol=tolerance):
            break
        else:
            cost_matrix = cost_matrix_current

    print(cost_matrix)
    return cost_matrix_current

        

    
def generate_state_pairs(trajectories_dir):

    trajectory_files = [f for f in os.listdir(trajectories_dir) if os.path.isfile(os.path.join(trajectories_dir, f))]
    unique_pairs = list(itertools.combinations(trajectory_files, 2))
    print(unique_pairs)
    for pair in unique_pairs:
        trajectory_file_1, trajectory_file_2 = os.path.join(trajectories_dir, pair[0]), os.path.join(trajectories_dir,pair[1])

        with open(trajectory_file_1, 'r') as trajectory_file_1:
            trajectories_env_1 = json.load(trajectory_file_1)
        with open(trajectory_file_2, 'r') as trajectory_file_2:
            trajectories_env_2 = json.load(trajectory_file_2)
        # if trajectories_env_1['context'] ==  trajectories_env_2['context']:
        #     continue
        trajectories_pair = trajectories_env_1['state_trajectory'], trajectories_env_2['state_trajectory']
        pi_star_file_path_pair = (trajectories_env_1['policy'], trajectories_env_2['policy'])
        
        cost_matrix = DIST(trajectories_pair, pi_star_file_path_pair, has_discrete_actions=True)
        print('this was', trajectories_env_1['context'], trajectories_env_2['context'], trajectories_env_1['context']- trajectories_env_2['context'])

        # print(cost_matrix.shape)
        # print(cost_matrix)

        

        # states_cartesian_product = list(itertools.product(states_env_1, states_env_2))
        # cost_matrix = np.array([])
        # for state_pair in states_cartesian_product:
        #     dist = DIST(state_pair, pi_star_file_path_pair)

        



    # for f in os.listdir(trajectories_dir):
    #     file_path = os.path.join(trajectories_dir,f)
    #     if os.path.isfile(file_path):
    #         with open(file_path):
    #             trajectory_file = json.load(file_path)
    #             env_context = trajectory_file['context']






def main():
    # uncomment below to generate trajectpries
    # context = {'gravity': 0.0025*1.8}
    # wrapper_context = {0:context}
    # base_env = CARLMountainCar(contexts=wrapper_context)
    # print('base',base_env.observation_space)
    # base_env_space = spaces.Box(shape=(2,), low=-np.inf, high=np.inf)
    # env= CARL_env_wrapper(base_env, base_env_space)
    # model_path = train_policy(env)
    # #model_path ='/Users/zasulta/Documents/DL/23_05_2024_DL_test/PPO/MountainCar-v0/0.0002_2024-05-24_19:11:43'
    # generate_trajectories(env, model_path)
    generate_state_pairs('trajectories/MountainCar-V0')



CONTEXT_VAR = 'gravity'
policies_save_dir = 'PPO'
trajectories_save_dir = 'trajectories'


if __name__ == "__main__":
    main()


# def DIST(env, pi_star):
#     #
#     action_space = env.action_space

#     if isinstance(action_space, spaces.Discrete):
        # TV