#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring to filter filenames."
    exit 1
fi

# Get the substring from the command-line argument
substring="$1"

script_dir=$(dirname "$0")
custom_data_dir="${script_dir}/custom_data/${substring}/"
train_script="${script_dir}/../abcboost_train"
predict_script="${script_dir}/../abcboost_predict"
test_data="${script_dir}/../data/${substring}.test.csv"

# --------------------------------
# 1) Source the mapping file
# --------------------------------
source "${script_dir}/dataset_task_map.sh"

# By default, if not found, set to "binary"
task_type="${dataset_task[$substring]:-binary}"
echo "Detected task type for '${substring}': $task_type"

# --------------------------------
# 2) Decide method & model suffix
# --------------------------------
case "$task_type" in
  "binary")
    # E.g., robustlogit
    method="robustlogit"
    model_suffix="robustlogit_J20_v0.1"
    ;;
  "multiclass")
    # E.g., mart
    method="mart"
    model_suffix="mart_J20_v0.1"
    ;;
  "regression")
    # E.g., regression
    method="regression"
    model_suffix="regression_J20_v0.1_p2"
    ;;
  *)
    echo "Unknown task type ($task_type). Must be 'binary', 'multiclass', or 'regression'."
    exit 1
    ;;
esac

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

        # --------------------------------
        # 3) Train command
        # --------------------------------
        # For example:
        #   -method robustlogit -lp 2 ...
        # or  -method mart ...
        # or  -method regression ...
        $train_script -method "$method" \
                      -lp 2 \
                      -data "$dataset" \
                      -J 20 \
                      -v 0.1 \
                      -iter 1000

        # Derive the model name from the dataset name plus the suffix
        # e.g., dataset_name.csv_robustlogit_J20_v0.1.model
        model_file="${dataset_name}.csv_${model_suffix}.model"

        # --------------------------------
        # 4) Predict command
        # --------------------------------
        $predict_script -data "$test_data" -model "$model_file"

        # Remove model file to optimize space
        rm *model && rm *prediction && rm *log
    else
        echo "Skipping dataset: $dataset (does not contain substring: $substring)"
    fi
done

mv *.model "${script_dir}/models" 2>/dev/null
mv *.prediction "${script_dir}/predictions" 2>/dev/null
mv *log "${script_dir}/logs" 2>/dev/null

echo "Training and prediction complete for all custom datasets containing substring: $substring."
