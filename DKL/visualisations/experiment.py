
from datetime import datetime
from ray import tune
from DKL.pb2 import PB2
#from ray.tune.examples.pbt_function import pbt_function
from ray.tune.schedulers.pbt import PopulationBasedTrainingReplay
from DKL.pbt_function import pbt_function
import os
import numpy as np
import ray
from ray import train, tune
import torch
import random
from DKL.PB2_DKL import PB2_dkl
import matplotlib.pyplot as plt
from ray.tune import sample_from
# run "pip install gpy" to use PB2



def run():
    ###########

    ws_dir = '/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/testing_dir'
    seed=0
    pertubation_interval=10
    num_samples = 4
    scheduler = 'pb2_dkl'
    num_iterations=50
    algo = 'repro_test_toy_func'

    ##########

    pb2_dkl = PB2_dkl(
        metric="mean_accuracy",
         time_attr="training_iteration",
        mode="max",
        perturbation_interval=pertubation_interval,
        hyperparam_bounds={"lr": [0.0001, 0.1]},
        synch=True, # for reproducability
        seed=seed
           
    )

    pb2 = PB2(
        metric="mean_accuracy",
         time_attr="training_iteration",
        mode="max",
        perturbation_interval=pertubation_interval,
        hyperparam_bounds={"lr": [0.0001, 0.1]},
        synch=True, # for reproducability
           
    )

    scheduler_map = {'pb2' : pb2, 'pb2_dkl': pb2_dkl}

    timelog = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    save_dir = os.path.join(ws_dir , "{}_{}_{}_Size_{}_{}".format(
        timelog,
        str(seed),
        scheduler,
        str(num_samples),
        algo
        
    ))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loguniform_dist = tune.loguniform(1e-5, 1e-3)
    samples = [loguniform_dist.sample() for _ in range(num_samples)]
    
    sample_iter = iter(samples)

# Define a lambda function to return the next unique sample
    get_sample = lambda: next(sample_iter)

    tuner = tune.Tuner(
        pbt_function,
        run_config=train.RunConfig(
            name=algo,
            storage_path=save_dir,
            verbose=False,
            stop={
                # Stop when done = True or at some # of train steps
                # (whichever comes first)
                "done": True,
                "training_iteration": num_iterations,
            },
            failure_config=train.FailureConfig(
                fail_fast=True,
            ),
            checkpoint_config=train.CheckpointConfig(
                checkpoint_score_attribute="mean_accuracy",
                #checkpoint_frequency=pertubation_interval,
            ),
        ),

        tune_config=tune.TuneConfig(
            scheduler=scheduler_map[scheduler],
            num_samples=num_samples,
            
        ),
        param_space={"lr":tune.sample_from(get_sample)  #1e-3 #sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
                     ,
                     'seed':seed, 
                     "checkpoint_interval": pertubation_interval},#{"lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand()))},
        
    )




    results_grid = tuner.fit()
    best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")

    # Print `path` where checkpoints are stored
    print('Best result path:', best_result.path)

    # Print the best trial `config` reported at the last iteration
    # NOTE: This config is just what the trial ended up with at the last iteration.
    # See the next section for replaying the entire history of configs.
    print("Best final iteration hyperparameter config:\n", best_result.config)

    # Plot the learning curve for the best trial
    df = best_result.metrics_dataframe
    # Deduplicate, since PBT might introduce duplicate data
    df = df.drop_duplicates(subset="training_iteration", keep="last")
    df.plot("training_iteration", "mean_accuracy")
    plt.xlabel("Training Iterations")
    plt.ylabel("Test Accuracy")
    plt.savefig(os.path.join(save_dir, 'best_run.png'))

def vis(run_directory):

    # List all files in the directory
    files = os.listdir(run_directory)

    # Filter out the text files
    text_files = [file for file in files if file.endswith('.txt')]
    ray_run =text_files[1]

    print(ray_run)
    

    replay = PopulationBasedTrainingReplay(os.path.join(run_directory,ray_run))

    print(replay.config)  # Initial config
    print(replay._policy)  # Schedule, in the form of tuples (step, config)

run()
# Define the lambda function to generate random values
# seed=1
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# def generate_sample(spec):
#     return random.uniform(1e-5, 1e-3)

# # Create a sample function with sample_from
# sample_fn = sample_from(generate_sample)

# # Print multiple samples

# # Define the loguniform distribution
# loguniform_dist = tune.loguniform(1e-5, 1e-3)

# # Sample from the distribution
# samples = [loguniform_dist.sample() for _ in range(5)]

# # Print the samples
# for sample in samples:
#     print(sample)  # Pass None if spec is not used
#run_dir='/Users/zasulta/ray_results/pbt_function_2024-07-25_11-47-15'
#run_dir='/Users/zasulta/ray_results/pbt_function_2024-07-25_12-27-36'
# vis(run_dir)
# Best final iteration hyperparameter config:
#  {'lr': 9.999999747378752e-05, 'seed': 0, 'checkpoint_interval': 10}
# 0.00010438750698400987
# 0.000939020130156147
# 0.0004057063380105475
# 0.00029642136384339996
# 0.0005892888174377769