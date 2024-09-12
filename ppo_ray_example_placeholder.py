import argparse
import json
import os
import random
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ray.tune.registry import register_env
import torch
from DKL.PB2_DKL import PB2_dkl
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2

from CARL_env_reg_wrapper import CARLWrapper


def dict_to_path_string(params):
    # Flatten the dictionary into key=value format
    elements = [f"{key}_{value}" for key, value in params.items()]
    
    # Join the elements with underscores or any other separator
    path_friendly_string = "_".join(elements)
    
    # Replace any problematic characters if needed
    path_friendly_string = path_friendly_string.replace(" ", "_").replace("{", "").replace("}", "").replace(":", "")
    
    return path_friendly_string

# Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config

def env_creator(env_name):
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
    return env





# def env_creator(env_config):
#     #base_env = CARLMountainCar(contexts=env_config)
#     #print('base',base_env.observation_space)
#     #base_env_space = spaces.Box(shape=(2,), low=-np.inf, high=np.inf)
#     #env= CARL_env_wrapper(base_env, base_env_space)
#     env = CARLWrapper(contexts={0:env_config})
#     return env  # return an env instance


# register_env("CARLMountainCar", env_creator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=1000000)
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--t_ready", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", type=str, default='{"gravity": 0.0025}')
    parser.add_argument(
        "--horizon", type=int, default=1600
    )  # make this 1000 for other envs
    parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
    parser.add_argument("--env_name", type=str, default="CARLMountainCar") #"CartPole-v1"
    parser.add_argument(
        "--criteria", type=str, default="timesteps_total"
    )  # "training_iteration", "time_total_s"
    parser.add_argument(
        "--net", type=str, default="32_32"
    )  # May be important to use a larger network for bigger tasks.
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--method", type=str, default="pb2_dkl")  # ['pbt', 'pb2']
    parser.add_argument("--save_csv", type=bool, default=True)
    parser.add_argument('--ws_dir', type=str, default=r'C:\Users\zaina\Downloads\Code\metapbt-2.0\metapbt2.0\testing')

    args = parser.parse_args()
    context = json.loads(args.context.replace("'", ""))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = env_creator(args.env_name)
    register_env(env_name, lambda: env_config, env(contexts={0:env_config}))

    print(args.env_name, args.context)

    # bipedalwalker needs 1600
    if args.env_name in ["BipedalWalker-v2", "BipedalWalker-v3"]:
        horizon = 1600
    else:
        horizon = 1000

    timelog = (
        str(datetime.date(datetime.now()))
    )

    args.dir = os.path.join(args.ws_dir , "{}_{}_{}_{}_Size{}_{}_{}".format(
        timelog,
        args.algo,
        dict_to_path_string(context),
        args.method,
        str(args.num_samples),
        args.env_name,
        args.criteria,
    ))
    pbt = PopulationBasedTraining(
        time_attr=args.criteria,
        metric='env_runners/episode_reward_mean',
        mode="max",
        perturbation_interval=args.t_ready,
        resample_probability=args.perturb,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.1, 0.5),
            "lr": lambda: random.uniform(1e-3, 1e-5),
            "train_batch_size": lambda: random.randint(1000, 60000),
        },
        custom_explore_fn=explore,
    )

    pb2_dkl=PB2_dkl(
        time_attr=args.criteria,
        metric='env_runners/episode_reward_mean',
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            #"lambda": [0.9, 1.0],
            #"clip_param": [0.1, 0.5],
            "lr": [1e-5, 1e-3],
            #"train_batch_size": [1000, 60000],
        },
        seed=args.seed,
        #save_path=args.dir
    
    )
    pb2 = PB2(
        time_attr=args.criteria,
        metric='env_runners/episode_reward_mean',
        mode="max",
        perturbation_interval=args.t_ready,
        quantile_fraction=args.perturb,  # copy bottom % with top %
        # Specifies the hyperparam search space
        hyperparam_bounds={
            #"lambda": [0.9, 1.0],
            #"clip_param": [0.1, 0.5],
            "lr": [1e-5, 1e-3],
            #"train_batch_size": [1000, 60000],
        },
    )

    methods = {"pbt": pbt, "pb2": pb2, 'pb2_dkl':pb2_dkl}





    analysis = run(
        args.algo,
        name="{}_{}_seed{}_{}".format(
            args.method, args.env_name, str(args.seed), dict_to_path_string(context)
        ),
        scheduler=methods[args.method],
        verbose=1,
        num_samples=args.num_samples,
        reuse_actors=False,
        stop={args.criteria: args.max},
        config={
            "env": args.env_name,
            "log_level": "INFO",
            "seed": args.seed,
            "kl_coeff": 1.0,
            "num_gpus": 0,
            "horizon": horizon,
            "observation_filter": "MeanStdFilter",
            "model": {
                "fcnet_hiddens": [
                    int(args.net.split("_")[0]),
                    int(args.net.split("_")[1]),
                ],
                #"free_log_std": True,
            },
            #"num_sgd_iter": 10,
            #"sgd_minibatch_size": 128,
            #"lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
            #"clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
            "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
            #"train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
            "env_config": context,
        },
        storage_path=args.dir
    )

    # Step 1: Concatenate all dataframes to find the best worker
    all_dfs = list(analysis.trial_dataframes.values())
    combined_df = pd.concat(all_dfs, axis=0)

    # Step 2: Identify the best worker based on the highest mean episode reward
    best_trial_idx = combined_df['env_runners/episode_reward_mean'].idxmax()
    best_trial = combined_df.loc[best_trial_idx]

    # Step 3: Extract the data for the best trial
    best_trial_data = combined_df[combined_df['trial_id'] == best_trial['trial_id']]

    if not (os.path.exists(os.path.join(args.dir,'data'))):
        os.makedirs(os.path.join(args.dir,'data'))

    # Step 4: Plot the mean episode reward against time
    plt.figure(figsize=(10, 6))
    plt.plot(best_trial_data['time_total_s'], best_trial_data['env_runners/episode_reward_mean'])
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Episode Reward')
    plt.title(f"Mean Episode Reward Over Time for Best Trial (ID: {best_trial['trial_id']})")
    plt.grid(True)
    plt.savefig(os.path.join(args.dir,'data','best_perf_{seed}.png'.format(seed=str(args.seed))))

    results = pd.DataFrame()
    for i in range(args.num_samples):
        df = all_dfs[i]
        df["Agent"] = i
        results = pd.concat([results, df]).reset_index(drop=True)

    if args.save_csv:
        save_path = os.path.join(args.dir,'data',"seed{}.csv".format( str(args.seed)))
        results.to_csv(save_path)
