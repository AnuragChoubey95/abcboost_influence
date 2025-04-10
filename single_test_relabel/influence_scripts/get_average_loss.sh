#!/usr/bin/env zsh

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a dataset substring."
    exit 1
fi

# Get the dataset substring from the command-line argument
substring="$1"

typeset -A dataset_task
dataset_task=(
    bank_marketing binary
    htru2 binary
    credit_card binary
    diabetes binary
    german binary
    spambase binary
    flight_delays binary
    no_show binary
    dry_bean multiclass
    adult multiclass
    concrete regression
    energy regression
    power_plant regression
    wine_quality regression
)

# 2) Lookup the task type
task_type="${dataset_task[$substring]}"
if [ -z "$task_type" ]; then
    echo "Warning: No task type found for '$substring'. Defaulting to binary."
    task_type="binary"
fi

echo "Calling get_average_loss.py with arguments: substring='$substring', task_type='$task_type'"

# 3) Call the Python script
python3 get_average_loss.py "$substring" "$task_type"
