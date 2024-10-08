#!/bin/bash

# Define the base directory containing the subdirectories
base_dir="/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/pb2_job_scripts/CARLLunarLander"

# Extract the environment name from the base directory path
env_name=$(basename "$base_dir")

# Extract the scheduler name from the base directory path
scheduler_name=$(basename "$(dirname "$base_dir")")

# Get the list of contexts from the arguments (if any)
contexts_to_run=("$@")

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

# Iterate through each subdirectory in the base directory
for subdir in "$base_dir"/*/; do
  # Extract the name of the subdirectory (a, b, c, etc.)
  feature_name=$(basename "$subdir")
  
  # Iterate through the .sh scripts in each subdirectory
  for script in "$subdir"context_*.sh; do
    # Extract the context value (1-22) from the script name
    context_value=$(basename "$script" | sed 's/context_\([0-9]\+\).sh/\1/')

    # Check if we should run this context based on the provided arguments
    if should_run_context "$context_value"; then
      # Generate the current date and time in the format YYYY-MM-DD_HH:MM.SS
      current_date=$(date +"%Y-%m-%d_%H:%M.%S")
      
      # Generate the job name following the format: <date>_<scheduler>.<env>.<feature>.<context>
      job_name="${current_date}_${scheduler_name}.${env_name}.${feature_name}.c${context_value}"
      
      # Run the job using job_wrapper
      ./job_wrapper.sh "$job_name" "$script"
      
      # Optional: Add a print statement to see what's running
      echo "Running job: $job_name with script: $script"
    fi
  done
done
