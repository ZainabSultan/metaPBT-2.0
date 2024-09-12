from ray.rllib.algorithms.ppo import PPOConfig
import numpy as np
import torch
import tensorflow as tf
from ray.tune.registry import register_env

# Set seeds
np.random.seed(0)
torch.manual_seed(0)
tf.random.set_seed(0)


def env_creator(env_config):
    print(env_config)
    env_name = env_config.pop('env_name')
    if env_name == 'CARLCartPole':
        from CARL_env_reg_wrapper_copy import CARLCartPoleWrapper as env
    elif env_name == 'CARLMountainCar':
        from CARL_env_reg_wrapper_copy import CARLMountainCarWrapper as env
    elif env_name == 'CARLMountainCarCont':
        from CARL_env_reg_wrapper_copy import CARLMountainCarContWrapper as env
    elif env_name == 'CARLAcrobot':
        from CARL_env_reg_wrapper_copy import CARLAcrobotWrapper as env
    elif env_name == 'CARLPendulum':
        from CARL_env_reg_wrapper_copy import CARLPendulumWrapper as env
    else:
        raise NotImplementedError
    return env(contexts={0:env_config})
register_env( 'CARLCartPole', env_creator)
config = PPOConfig()
# Activate new API stack.
config.environment("CARLCartPole")
config.seed(0)
config.env_runners(num_env_runners=1)
config.training(
    gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256
)

# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
for i in range(50):
    result = algo.train()
    print(f"Iteration {i}: {result}")

# Save results for comparison
import json
with open("/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir/training_results.json", "w") as f:
    json.dump(result, f)