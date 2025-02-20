#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring to filter filenames."
    exit 1
fi

# Get the substring from the command-line argument
substring="$1"

# Paths and variables
script_dir=$(dirname "$0")
custom_data_dir="${script_dir}/custom_data/${substring}/"
train_script="${script_dir}/../abcboost_train"
predict_script="${script_dir}/../abcboost_predict"
test_data="${script_dir}/../data/${substring}.test.csv"

# Check if custom_data_dir exists and has files
if [ ! -d "$custom_data_dir" ]; then
    echo "Error: Directory $custom_data_dir does not exist."
    exit 1
fi

if [ -z "$(ls -A ${custom_data_dir}*.csv 2>/dev/null)" ]; then
    echo "Error: No matching files found in $custom_data_dir for substring: $substring"
    exit 1
fi

# Ensure necessary directories exist
mkdir -p "${script_dir}/models"
mkdir -p "${script_dir}/predictions"
mkdir -p "${script_dir}/logs"

# Loop through each custom dataset in the custom data directory
for dataset in ${custom_data_dir}*.csv; do
    # Check if the dataset name contains the provided substring
    if [[ $(basename "$dataset") == *"$substring"* ]]; then
        echo "Processing dataset: $dataset"

        # Extract dataset name without path and extension
        dataset_name=$(basename "$dataset" .csv)

        # Train the model on the custom dataset
        $train_script -method robustlogit -lp 2 -data "$dataset" -J 20 -v 0.1 -iter 1000

        # # Derive the model name from the dataset name
        model_file="${dataset_name}.csv_robustlogit_J20_v0.1.model"

        # # Predict using the trained model
        $predict_script -data "$test_data" -model "$model_file"
        
        # #Remove model file to optimize space
        rm *model && rm *prediction && rm *log
    else
        echo "Skipping dataset: $dataset (does not contain substring: $substring)"
    fi
done

mv *.model "${script_dir}/models"
mv *.prediction "${script_dir}/predictions"
mv *log "${script_dir}/logs"

echo "Training and prediction complete for all custom datasets containing substring: $substring."
