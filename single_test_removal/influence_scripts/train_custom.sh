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
custom_data_dir="${script_dir}/../custom_data/${substring}/"
train_script="${script_dir}/../../abcboost_train"
predict_script="${script_dir}/../../abcboost_predict"

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
        # Extract the column index from the training file name (robust version)
        column_idx=$(echo "$dataset_name" | sed -E 's/.*_column([0-9]+).*/\1/')

        if [ -z "$column_idx" ]; then
            echo "Error: Could not extract column index from dataset name: $dataset_name"
            exit 1
        fi

        # Build the test sample path
        test_data="${script_dir}/../test_data/${substring}/${substring}_test_column_${column_idx}.csv"

        if [ ! -f "$test_data" ]; then
            echo "Error: Test data file not found: $test_data"
            exit 1
        fi



        # Parse task type from dataset_task_map.sh
        task_type=$(grep "\[$substring\]" "${script_dir}/dataset_task_map.sh" | grep -o '".*"' | tr -d '"')

        # Determine the method and model suffix
        case "$task_type" in
            binary)
                method="robustlogit"
                model_suffix="csv_robustlogit_J20_v0.1.model"
                ;;
            multiclass)
                method="mart"
                model_suffix="csv_mart_J20_v0.1.model"
                ;;
            regression)
                method="regression"
                model_suffix="csv_regression_J20_v0.1_p2.model"
                ;;
            *)
                echo "Error: Unknown task type '$task_type' for substring '$substring'"
                exit 1
                ;;
        esac

        echo "'$method'"

        # Train the model using the appropriate method
        $train_script -method "$method" -lp 2 -data "$dataset" -J 20 -v 0.1 -iter 1000

        # Derive the model file name based on method
        model_file="${dataset_name}.${model_suffix}"


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
