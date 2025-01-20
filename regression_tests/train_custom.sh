#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring to filter filenames."
    exit 1
fi

# Get the substring from the command-line argument
substring="$1"

# Paths and variables
train_script=".././abcboost_train"
predict_script=".././abcboost_predict"
custom_data_dir="custom_data/${substring}/"
test_data="../data/${substring}.test.csv"

# Loop through each custom dataset in the custom data directory
for dataset in ${custom_data_dir}*.csv; do
    # Check if the dataset name contains the provided substring
    if [[ $(basename "$dataset") == *"$substring"* ]]; then
        echo "Processing dataset: $dataset"
        # sleep 3  # Pause for 3 seconds

        # Extract dataset name without path and extension
        dataset_name=$(basename "$dataset" .csv)

        # Train the model on the custom dataset
        $train_script -method regression -lp 2 -data "$dataset" -J 20 -v 0.1 -iter 1000

        # Derive the model name from the dataset name
        model_file="${dataset_name}.csv_regression_J20_v0.1_p2.model"

        # Predict using the trained model
        $predict_script -data "$test_data" -model "$model_file"
    else
        echo "Skipping dataset: $dataset (does not contain substring: $substring)"
    fi
done

# Move all files with suffix ".model" into dir models
mv *.model models/
# Move all files with suffix ".prediction" into dir predictions
mv *.prediction predictions/
# Move all files with suffix "log" into dir logs
mv *log logs/

echo "Training and prediction complete for all custom datasets containing substring: $substring."
