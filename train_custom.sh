#!/bin/bash

# Paths and variables
train_script="./abcboost_train"
predict_script="./abcboost_predict"
custom_data_dir="custom_data/"
test_data="data/comp_cpu.test.csv"

# Ensure necessary directories exist
mkdir -p models
mkdir -p predictions
mkdir -p logs

# Loop through each custom dataset in the custom data directory
for dataset in ${custom_data_dir}*.csv; do
    echo "Processing dataset: $dataset"

    # Extract dataset name without path and extension
    dataset_name=$(basename "$dataset" .csv)

    # Train the model on the custom dataset
    $train_script -method regression -lp 2 -data "$dataset" -J 20 -v 0.1 -iter 1000

    # Derive the model name from the dataset name
    model_file="${dataset_name}.csv_regression_J20_v0.1_p2.model"

    # Predict using the trained model
    $predict_script -data "$test_data" -model "$model_file"
done

# Move all files with suffix ".model" into dir models
mv *.model models/
# Move all files with suffix ".prediction" into dir predictions
mv *.prediction predictions/
# Move all files with suffix "log" into dir logs
mv *log logs/

echo "Training and prediction complete for all custom datasets."
