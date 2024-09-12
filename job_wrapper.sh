#!/bin/bash

# Check if the correct number of arguments is passed
if [ $# -lt 2 ]; then
    echo "Usage: $0 <job_name> <job_script_path>"
    exit 1
fi

JOB_NAME=$1
JOB_SCRIPT=$2

# Create the output destination directory
mkdir -p $(ws_find dkl_exps)/cluster_logs/${JOB_NAME}/

# Submit the job with the provided script
JOB_ID=$(sbatch --job-name=${JOB_NAME} ${JOB_SCRIPT})

echo $JOB_ID