#!/bin/zsh

# Define dataset-task map using associative array
declare -A dataset_task=(
    [bank_marketing]="binary"
    [htru2]="binary"
    [credit_card]="binary"
    [diabetes]="binary"
    [german]="binary"
    [spambase]="binary"
    [flight_delays]="binary"
    [no_show]="binary"

    [dry_bean]="multiclass"
    [adult]="multiclass"

    [concrete]="regression"
    [energy]="regression"
    [power_plant]="regression"
    [wine_quality]="regression"
    [life_expectancy]="regression"
)

# Function to get task type for a dataset
get_task_type() {
    local dataset=$1
    echo "${dataset_task[$dataset]:-unknown}"
}

# Loop through each dataset and execute scripts
for dataset in "${(@k)dataset_task}"; do
    echo "Processing dataset: $dataset"
    task_type=$(get_task_type "$dataset")
    echo "Task type: $task_type"

    python3 split_and_rank.py "$dataset"
    python3 create_training_data.py "$dataset"
    python3 train_test.py "$dataset"
    python3 get_change_loss.py "$dataset"

    echo "Completed execution for dataset: $dataset"
    echo "----------------------------------------"

    # Cleanup generated files
    rm -f *.model
    rm -f *log
    rm -f *prediction
done
