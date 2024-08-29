

from ray import tune
from pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function
from ray.tune.schedulers.pbt import PopulationBasedTrainingReplay
import os
import numpy as np
import ray
from ray import train, tune
import torch
import random
from PB2_DKL import PB2_dkl
import matplotlib.pyplot as plt

# run "pip install gpy" to use PB2
smoke_test=True



def run():
    seed=0
    pertubation_interval=5
    pb2 = PB2(
        metric="mean_accuracy",
         time_attr="training_iteration",
        mode="max",
        perturbation_interval=pertubation_interval,
        hyperparam_bounds={"lr": [0.0001, 0.1]},
        synch=True, # for reproducability
        seed=seed
        
        
    )
    tuner = tune.Tuner(
        pbt_function,
        run_config=train.RunConfig(
            name="pbt_function_DKLPB2",
            storage_path="/Users/zasulta/Documents/DL/23_05_2024_DL_test/DKL/visualisations/ray_results",
            verbose=False,
            stop={
                # Stop when done = True or at some # of train steps
                # (whichever comes first)
                "done": True,
                "training_iteration": 20 if smoke_test else 1000,
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
            scheduler=pb2,
            num_samples=4,
            
        ),
        param_space={"lr": 0.0001, "checkpoint_interval": pertubation_interval},#{"lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand()))},
        
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
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
    plt.show()

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
#run_dir='/Users/zasulta/ray_results/pbt_function_2024-07-25_11-47-15'
#run_dir='/Users/zasulta/ray_results/pbt_function_2024-07-25_12-27-36'
# vis(run_dir)