#!/bin/bash

# Take the base directory as input
#base_dir="/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/meta_dkl_scripts/CARLCartPole/length"
#base_dir='/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/gen_meta_dkl_scripts/CARLMountainCar/gravity'
#base_dir='/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/gen_meta_dkl_scripts/CARLLunarLander/LEG_H'
base_dir='/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/pb2_job_scripts/CARLLunarLander/GRAVITY_Y'
# Extract the environment name from the directory above the feature directory
env_name=$(basename "$(dirname "$(dirname "$base_dir")")")
env_name=$(basename "$(dirname "$base_dir")")

# Extract the feature name from the base directory path
feature_name=$(basename "$(dirname "$base_dir")")
feature_name=$(basename $base_dir)

# Extract the scheduler name from the directory above the environment directory
scheduler_name=$(basename  "$(dirname "$(dirname "$base_dir")")")

# Get the list of contexts from the arguments (if any), starting from the second argument
contexts_to_run=("${@:2}")

# Function to check if a context is in the list of contexts to run
should_run_context() {
  local context_value="$1"
  
  # If no contexts are provided, run all
  if [ ${#contexts_to_run[@]} -eq 0 ]; then
    return 0
  fi
  
  # Check if the current context is in the list of contexts to run
  for context in "${contexts_to_run[@]}"; do
    if [ "$context" == "c$context_value" ]; then
      return 0
    fi
  done
  return 1
}

# Iterate through each .sh script in the provided directory
for script in "$base_dir"/context_*.sh; do
  # Extract the context value (1-22) from the script name
  context_value=$(basename "$script" | sed 's/context_\([0-9]\+\)_seed_.*\.sh/\1/')

  # Check if we should run this context based on the provided arguments
  if should_run_context "$context_value"; then
    # Generate the job name without .sh by removing the extension
    job_name="meta_dkl_${env_name}_${feature_name}_${context_value}"
    
    # Run the job using job_wrapper
    ./job_wrapper.sh "$job_name" "$script"
    
    # Print statement to show what's being run
    echo "$job_name"
    echo "Running job: $job_name with script: $script"
  fi
done
