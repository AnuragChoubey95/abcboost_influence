#!/bin/zsh

# Define dataset-task map using arrays
dataset_names=("compas" "htru2" "credit_card" "flight_delays" "diabetes" "no_show" "german_credit" "spambase" "surgical" "vaccine"
               "dry_bean"
               "concrete" "energy_efficiency" "life_expectancy" "naval" "combined_cycle_power_plant" "wine_quality")

dataset_tasks=("binary" "binary" "binary" "binary" "binary" "binary" "binary" "binary" "binary" "binary"
               "multiclass"
               "regression" "regression" "regression" "regression" "regression" "regression")

# Function to get task type for a dataset
get_task_type() {
    local dataset=$1
    for i in {1..${#dataset_names[@]}}; do
        if [[ "${dataset_names[$i]}" == "$dataset" ]]; then
            echo "${dataset_tasks[$i]}"
            return
        fi
    done
    echo "unknown"
}

# Loop through each dataset and execute scripts
for dataset in "${dataset_names[@]}"; do
    echo "Processing dataset: $dataset"

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
