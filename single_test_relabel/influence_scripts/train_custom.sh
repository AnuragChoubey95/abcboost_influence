#!/usr/bin/env zsh

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring to filter filenames."
    exit 1
fi

# Get the substring from the command-line argument
substring="$1"

script_dir=$(dirname "$0")
custom_data_dir="${script_dir}/../custom_data/${substring}/"
train_script="${script_dir}/../../abcboost_train"
predict_script="${script_dir}/../../abcboost_predict"

# --------------------------------
# 1) Declare dataset → task mapping
# --------------------------------
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
    concrete regression
    energy regression
    power_plant regression
    wine_quality regression
)

# --------------------------------
# 2) Decide method & model suffix
# --------------------------------
task_type="${dataset_task[$substring]}"
if [ -z "$task_type" ]; then
    echo "Warning: No task type found for $substring. Defaulting to binary."
    task_type="binary"
fi
echo "Detected task type for '${substring}': $task_type"

case "$task_type" in
  "binary")
    method="robustlogit"
    model_suffix="robustlogit_J20_v0.1"
    ;;
  "multiclass")
    method="mart"
    model_suffix="mart_J20_v0.1"
    ;;
  "regression")
    method="regression"
    model_suffix="regression_J20_v0.1_p2"
    ;;
  *)
    echo "Unknown task type ($task_type). Must be 'binary', 'multiclass', or 'regression'."
    exit 1
    ;;
esac

# --------------------------------
# 3) Check if data dir exists
# --------------------------------
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

# --------------------------------
# 4) Loop over datasets and run train + predict
# --------------------------------
for dataset in ${custom_data_dir}*.csv; do
    if [[ $(basename "$dataset") == *"$substring"* ]]; then
        echo "Processing dataset: $dataset"

        # Extract dataset name
        dataset_name=$(basename "$dataset" .csv)

        # Train
        $train_script -method "$method" \
                      -lp 2 \
                      -data "$dataset" \
                      -J 20 \
                      -v 0.1 \
                      -iter 1000

        # Model name
        model_file="${dataset_name}.csv_${model_suffix}.model"

        # Extract column index (e.g., from '...column1442...')
        column_idx=$(echo "$dataset_name" | sed -E 's/.*_column([0-9]+).*/\1/')

        if [ -z "$column_idx" ]; then
            echo "Error: Could not extract column index from $dataset_name"
            exit 1
        fi

        # Build path to the corresponding test row
        test_data_path="${script_dir}/../test_data/${substring}/${substring}_test_column_${column_idx}.csv"

        if [ ! -f "$test_data_path" ]; then
            echo "Error: Test file not found for column $column_idx: $test_data_path"
            exit 1
        fi

        # Predict on the single test row
        $predict_script -data "$test_data_path" -model "$model_file"

        # Cleanup
        rm -f *model *prediction *log
    else
        echo "Skipping dataset: $dataset (does not contain substring: $substring)"
    fi
done

# Move files to directories
mv *.model "${script_dir}/models" 2>/dev/null
mv *.prediction "${script_dir}/predictions" 2>/dev/null
mv *log "${script_dir}/logs" 2>/dev/null

echo "Training and prediction complete for all custom datasets containing substring: $substring."