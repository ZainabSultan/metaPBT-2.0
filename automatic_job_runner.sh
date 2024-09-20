#!/bin/bash

# Define the base directory containing the subdirectories
base_dir="/home/fr/fr_fr/fr_zs53/DKL/metaPBT-2.0/pb2_job_scripts/holder"

# Iterate through each subdirectory in the base directory
for subdir in "$base_dir"/*/; do
  # Extract the name of the subdirectory (a, b, c, etc.)
  feature_name=$(basename "$subdir")
  
  # Iterate through the .sh scripts in each subdirectory
  for script in "$subdir"context_*.sh; do
    # Extract the context value (1-22) from the script name
    context_value=$(basename "$script" | sed 's/context_\([0-9]\+\).sh/\1/')
    
    # Generate the job name following the format: pb2.<feature>.<context>
    job_name="pb2.${feature_name}.c${context_value}"
    
    # Run the job using job_wrapper
    ./job_wrapper.sh "$job_name" "$script"
    
    # Optional: Add a print statement to see what's running
    echo "Running job: $job_name with script: $script"
  done
done
