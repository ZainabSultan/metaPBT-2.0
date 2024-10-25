import copy
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import gym
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from CARLWrapper import env_creator
import json

# Function to create environments for SubprocVecEnv
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

# Function to collect epsilon-greedy trajectories
def collect_epsilon_greedy_trajectories(model, env_config, n_steps, epsilon=0.2, nsteps_trained_policy=''):
    
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config
    env = env_creator(env_config=copy.deepcopy(env_config))
    if isinstance(model, str):
        model = PPO.load(model, env=env)
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    
    # Data to store the state, action, reward, next_state, and other information
    trajectories = []
    action_rng = np.random.RandomState(seed=42) 

    for _ in range(n_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            if isinstance(vec_env.action_space, gym.spaces.Discrete):
                action = action_rng.choice(vec_env.action_space.n)
            elif isinstance(vec_env.action_space, gym.spaces.Box):
                action = action_rng.uniform(low=vec_env.action_space.low, high=vec_env.action_space.high, size=vec_env.action_space.shape)
        else:
            action, _states = model.predict(obs, deterministic=True)

        next_obs, reward, done, info = vec_env.step(action)
        flattened_state = obs.flatten() if isinstance(obs, np.ndarray) else obs
        flattened_next_state = next_obs.flatten() if isinstance(next_obs, np.ndarray) else next_obs

        row = {}

        # Add state, action, next_state, reward, done, and context information to the row
        for i, state_value in enumerate(flattened_state):
            row[f'state_{i}'] = state_value
        if isinstance(action, np.ndarray):
            for i, action_value in enumerate(action):
                row[f'action_{i}'] = action_value
        else:
            row['action'] = action
        for i, next_state_value in enumerate(flattened_next_state):
            row[f'next_state_{i}'] = next_state_value

        row['reward'] = reward
        row['done'] = done
        row['env_context'] = env_context

        trajectories.append(row)

        if done[0]:
            obs = vec_env.reset()
        else:
            obs = next_obs

    df = pd.DataFrame(trajectories)

    state_columns = [col for col in df.columns if col.startswith('state_')]
    next_state_columns = [col for col in df.columns if col.startswith('next_state_')]
    action_columns = [col for col in df.columns if col.startswith('action_')]
    other_columns = [col for col in df.columns if col not in state_columns + next_state_columns + action_columns]

    df = df[state_columns + action_columns + next_state_columns + other_columns]

    df.to_csv(f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/datasets/{env_name}/{str(env_context)}.csv', sep=';')

    return df

# Function to train a policy with SubprocVecEnv
def train_policy(env_config, n_steps_train=1_000_000, num_envs=10):
    """
    Train a PPO policy using vectorized environments for faster training.
    `n_steps_train` defines the total timesteps for training.
    `num_envs` defines how many environments will run in parallel.
    """
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config
    
    # Create vectorized environments using SubprocVecEnv
    envs = SubprocVecEnv([make_env(env_config, i) for i in range(num_envs)], start_method='spawn')

    # Create PPO model with SubprocVecEnv and batch_size adjusted for vectorized training
    model = PPO(MlpPolicy, envs, verbose=2, n_steps=n_steps_train // num_envs, batch_size=100)

    # Evaluate the model before training
    mean_reward_before_train, std_reward_before_train = evaluate_policy(model, envs, n_eval_episodes=10)
    print(f"mean_reward before train: {mean_reward_before_train:.2f} +/- {std_reward_before_train:.2f}")

    # Train the PPO model
    model.learn(total_timesteps=n_steps_train, progress_bar=True)
    
    # Save the model after training
    model_path = f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trained_policies/{env_name}/{str(env_context)}_{n_steps_train}'
    model.save(model_path)
    print('Model saved')

    # Evaluate the model after training
    #mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    #print(f"mean_reward after train: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model

if __name__ == "__main__":
    """
    Main method to parse command-line arguments and run the training.
    """
    parser = argparse.ArgumentParser(description="Train PPO with vectorized environments")

    # Define command-line arguments
    parser.add_argument('--n_steps_train', type=int, default=1_000_000, 
                        help="Total number of timesteps to train the agent (default: 1M)")
    parser.add_argument('--n_steps_collect', type=int, default=1_000, 
                        help="Number of steps collected per environment per update (default: 1M)")
    parser.add_argument('--num_envs', type=int, default=8, 
                        help="Number of environments to run in parallel (default: 8)")
    parser.add_argument('--env_name', type=str, default='CARLMountainCar',
                        help="Name of the environment to train on")
    parser.add_argument("--context", type=str, default='{"gravity":0.00035}')

    # Parse the arguments
    args = parser.parse_args()

    epsilon = 0.2  # Epsilon for epsilon-greedy

    env_config = json.loads(args.context.replace("'", ""))
    env_config = env_config | {'env_name': args.env_name}
    
    # Train the policy using the provided args
    # trained_policy = train_policy(env_config=env_config, 
    #                               n_steps_train=args.n_steps_train, 
    #                               num_envs=args.num_envs)
    num_envs = args.num_envs
    n_steps_train = args.n_steps_train
    config = copy.deepcopy(env_config)
    env_name = config.pop('env_name')
    env_context = config
    
    # Create vectorized environments using SubprocVecEnv
    envs = SubprocVecEnv([make_env(env_config, i) for i in range(num_envs)], start_method='forkserver')

    # Create PPO model with SubprocVecEnv and batch_size adjusted for vectorized training
    model = PPO(MlpPolicy, envs, verbose=2, n_steps=n_steps_train // num_envs, batch_size=100)

    # Evaluate the model before training
    mean_reward_before_train, std_reward_before_train = evaluate_policy(model, envs, n_eval_episodes=10)
    print(f"mean_reward before train: {mean_reward_before_train:.2f} +/- {std_reward_before_train:.2f}")

    # Train the PPO model
    model.learn(total_timesteps=n_steps_train, progress_bar=True)
    
    # Save the model after training
    model_path = f'/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/DKL/EPI/trained_policies/{env_name}/{str(env_context)}_{n_steps_train}'
    model.save(model_path)
    print('Model saved')
    
    

    
    # Collect epsilon-greedy trajectories with the trained policy
    #collect_epsilon_greedy_trajectories(trained_policy, env_config, args.n_steps_collect, epsilon)
