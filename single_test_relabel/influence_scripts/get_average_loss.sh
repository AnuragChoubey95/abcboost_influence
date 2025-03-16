#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a dataset substring."
    exit 1
fi

# Get the dataset substring from the command-line argument
substring="$1"

# 1) Source the map
source dataset_task_map.sh

# 2) Lookup the task type from the mapping
task_type="${dataset_task[$substring]:-binary}"  # Default to 'binary' if not found

echo "Calling get_average_loss.py with arguments: substring='$substring', task_type='$task_type'"

# 3) Call the Python script with both arguments
python3 get_average_loss.py "$substring" "$task_type"
