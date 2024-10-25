



import copy
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import gym
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from CARLWrapper import env_creator
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import json
#from stable_baselines3.bench import Monitor
# New function to create vectorized environments
def make_env(env_config, rank, seed=0):
    """
    Utility function to create multiple instances of the same environment
    for SubprocVecEnv. Each instance gets a different seed.
    """
    def _init():
        env = env_creator(copy.deepcopy(env_config))
        env.seed = seed + rank
        return env
    return _init


def collect_epsilon_greedy_trajectories(model, env_config, n_steps, epsilon=0.2, nsteps_trained_policy=''):
    
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config
    env = env_creator(env_config=copy.deepcopy(env_config))
    if isinstance(model, str):
        model = PPO.load(model, env=env)
    # from pprint import pprint
    # pprint(vars(model))
    vec_env = model.get_env()
    obs = vec_env.reset()
    # Data to store the state, action, reward, next_state, and other information
    trajectories = []
    action_rng = np.random.RandomState(seed=42) 
    for _ in range(n_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            if isinstance(vec_env.action_space, gym.spaces.Discrete):
            # For Discrete action spaces, you sample from a set number of discrete actions
                action = action_rng.choice(vec_env.action_space.n)
            elif isinstance(vec_env.action_space, gym.spaces.Box):
                # For Box action spaces, use the `low` and `high` attributes
                action = action_rng.uniform(low=vec_env.action_space.low, high=vec_env.action_space.high, size=vec_env.action_space.shape)
        else:
            # Get the action from the policy model with probability (1 - epsilon)
            action, _states = model.predict(obs, deterministic=True)

        # Take a step in the environment
        next_obs, reward, done, info = vec_env.step(action)
        
        # Flatten state and next_state for easier column access
        flattened_state = obs.flatten() if isinstance(obs, np.ndarray) else obs
        flattened_next_state = next_obs.flatten() if isinstance(next_obs, np.ndarray) else next_obs

        # Collect information in the trajectories
        row = {}

        # Add state variables first
        for i, state_value in enumerate(flattened_state):
            row[f'state_{i}'] = state_value

        # Add action after state columns
        if isinstance(action, np.ndarray):  # For continuous action space
            for i, action_value in enumerate(action):  # action[0] because batch_size=1
                row[f'action_{i}'] = action_value
        else:  # For discrete action space, it's just a scalar
            row['action'] = action
        
        # Add next_state variables after action
        for i, next_state_value in enumerate(flattened_next_state):
            row[f'next_state_{i}'] = next_state_value

                # Add info dictionary as separate columns (if available) at the end
        # for key, value in info[0].items():
        #     row[key] = value

        # Add reward and done
        row['reward'] = reward
        row['done'] = done
        row['env_context'] = env_context
        # Add info dictionary as separate columns (if available) at the end

        trajectories.append(row)
        #print(row)
        #print(done)

        # If the episode is done (termination or truncation), reset the environment
        if done:
            obs = vec_env.reset()
        else:
            obs = next_obs  # Move to the next state
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(trajectories)

    # Rearrange the columns in the desired order: state -> action -> next_state -> info
    state_columns = [col for col in df.columns if col.startswith('state_')]
    next_state_columns = [col for col in df.columns if col.startswith('next_state_')]
    action_columns = [col for col in df.columns if col.startswith('action_')]
    other_columns = [col for col in df.columns if col not in state_columns + next_state_columns + action_columns]

    df = df[state_columns + action_columns + next_state_columns + other_columns]

    df.to_csv(f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/datasets/{env_name}/{str(env_context)}.csv', sep=';')


    return df





def train_policy(env_config, n_steps_train=1_000_000, num_envs=8):
    """
    Train a PPO policy using vectorized environments for faster training.
    `n_steps_train` defines the total timesteps for training.
    `n_steps_collect` defines the number of steps collected per update.
    `num_envs` defines how many environments will run in parallel.
    """
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config
    
    # Create vectorized environments using SubprocVecEnv
    #envs = SubprocVecEnv([make_env(env_config, i) for i in range(num_envs)])
    env = make_env(env_config, 0)()

    # Create PPO model with custom n_steps
    model = PPO(MlpPolicy, env, verbose=2, n_steps=n_steps_train, batch_size=100)

    # Evaluate the model before training
    mean_reward_before_train, std_reward_before_train = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward before train: {mean_reward_before_train:.2f} +/- {std_reward_before_train:.2f}")

    # Train the PPO model
    model.learn(total_timesteps=n_steps_train, progress_bar=True)
    n_total_steps = n_steps_train#*num_envs
    # Save the model
    print('done training!!')
    model_path = f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trained_policies/{env_name}/{str(env_context)}_{n_total_steps}'
    model.save(model_path)
    print('model saved')

    env = make_env(env_config, 0)()
    # Evaluate the model after training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward after train: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


if __name__ == "__main__":
    """
    Main method to parse command-line arguments and run the training.
    """
    parser = argparse.ArgumentParser(description="Train PPO with vectorized environments")

    # Define command-line arguments
    parser.add_argument('--n_steps_train', type=int, default=1000_000, 
                        help="Total number of timesteps to train the agent (default: 1M)")
    parser.add_argument('--n_steps_collect', type=int, default=1000, 
                        help="Number of steps collected per environment per update (default: 100K)")
    parser.add_argument('--num_envs', type=int, default=16, 
                        help="Number of environments to run in parallel (default: 8)")
    parser.add_argument('--env_name', type=str, default='CARLMountainCar',
                        help="Name of the environment to train on")
    parser.add_argument("--context", type=str, default='{"gravity":0.00035}')

    # Parse the arguments
    args = parser.parse_args()

    # Example environment configuration based on parsed args

    epsilon = 0.2  # Epsilon for epsilon-greedy

    env_config = json.loads(args.context.replace("'", ""))
    env_config= env_config | {'env_name':args.env_name}
    #trained_policy = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trained_policies/PPO_CARLBipedalWalker_2048_{\'TERRAIN_LENGTH\': 0.05}'
    

    # Train the policy using the provided args
    
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config

    n_steps_train=args.n_steps_train
    num_envs=args.num_envs
    trained_policy = train_policy(env_config=env_config, 
                 n_steps_train=args.n_steps_train, 
                 num_envs=args.num_envs)
    print('back from training')
    #trained_policy = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trained_policies/PPO_CARLBipedalWalker_4000_{\'TERRAIN_LENGTH\': 200}'
    collect_epsilon_greedy_trajectories(trained_policy, env_config, args.n_steps_collect, epsilon, args.n_steps_train)

# you train a policy, only provide the env config
# execute trajectory collection on the trained policy. you can pass a string of where the policy was stored
# ./job_wrapper.sh test_policy /home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/epi_scripts/test.sh



