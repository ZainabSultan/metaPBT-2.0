import gym
from stable_baselines3 import PPO
from ray import tune
from DKL.pb2 import PB2
import numpy as np

def train_ppo(config):
    env = gym.make(config["env_name"])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gae_lambda=config["gae_lambda"],
        gamma=config["gamma"],
        verbose=1,
    )
    model.learn(total_timesteps=config["total_timesteps"])
    mean_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)
    #tune.report(mean_reward=mean_reward)  # Report mean reward for tuning
    return {"mean_reward": mean_reward}
    

def evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False):
    all_episode_rewards = []
    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))

    mean_reward = np.mean(all_episode_rewards)
    return mean_reward

# Set up PB2 scheduler
pb2 = PB2(
    time_attr="training_iteration",
    reward_attr="mean_reward",
    perturbation_interval=5,
    hyperparam_bounds={
        "learning_rate": [1e-5, 1e-1],
    }
)

# Define configuration space
config = {
    "env_name": "CartPole-v1",
    "learning_rate": tune.uniform(1e-5, 1e-1),
    "n_steps": 2048,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "total_timesteps": 10000,
}

# Run the tuning
analysis = tune.run(
    train_ppo,
    config=config,
    num_samples=10,
    scheduler=pb2,
    progress_reporter=tune.CLIReporter(
        parameter_columns=["learning_rate"],
        metric_columns=["mean_reward", "training_iteration"]
    )
)

print("Best hyperparameters found were: ", analysis.best_config)

