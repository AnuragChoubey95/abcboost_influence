#!/usr/bin/env zsh

# This script automatically ranks training samples for a given dataset substring
# and removes top-k influential samples at various percentages.

# Usage: ./automate_ranking_removal.sh <substring>

# 1) Check arguments
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring."
    exit 1
fi

# 2) Get the substring from command-line
substring="$1"

# 3) Source the mapping from a separate file
#    which sets an associative array "dataset_task"
source ./dataset_task_map.sh

# If your dataset name is not found, default to "binary"
task_type="${dataset_task[$substring]:-binary}"
echo "Detected task type for '${substring}' => ${task_type}"

# 4) Determine the number of columns in the influence file for BoostIn
#    (assuming typical naming pattern for "binary", "multiclass", etc.)
boostin_file=""
lca_file=""

# Decide the correct influence filenames based on $task_type
if [ "$task_type" = "binary" ]; then
    boostin_file="../../influence_scores/${substring}.test.csv_robustlogit_J20_v0.1BoostIn_Influence.csv"
    lca_file="../../influence_scores/${substring}.test.csv_robustlogit_J20_v0.1LCA_Influence.csv"
elif [ "$task_type" = "multiclass" ]; then
    boostin_file="../../influence_scores/${substring}.test.csv_mart_J20_v0.1BoostIn_Influence.csv"
    lca_file="../../influence_scores/${substring}.test.csv_mart_J20_v0.1LCA_Influence.csv"
elif [ "$task_type" = "regression" ]; then
    boostin_file="../../influence_scores/${substring}.test.csv_regression_J20_v0.1_p2BoostIn_Influence.csv"
    lca_file="../../influence_scores/${substring}.test.csv_regression_J20_v0.1_p2LCA_Influence.csv"
else
    echo "Unknown task type ($task_type). Must be 'binary', 'multiclass', or 'regression'."
    exit 1
fi

echo "Using BoostIn file: $boostin_file"
echo "Using LCA file: $lca_file"

# 5) Verify the first file exists before reading columns
if [ ! -f "$boostin_file" ]; then
    echo "Error: $boostin_file not found. Check your naming or paths."
    exit 1
fi

num_columns=$(head -n 1 "$boostin_file" | awk -F',' '{print NF}')
if [ -z "$num_columns" ] || [ "$num_columns" -eq 0 ]; then
    echo "Error: Unable to determine the number of columns in the influence file ($boostin_file)."
    exit 1
fi

echo "Number of columns in the influence files: $num_columns"


# Generate random unique indices between 0 and (num_columns - 1) with seed=42
indices=($(python3 -c "import random; random.seed(42); print(' '.join(map(str, random.sample(range(0, $num_columns), 100))))"))


# 7) Define percentages
percentages=(0.1 0.5 1 1.5 2)

# 8) For each index => rank with both BoostIn & LCA => remove data at percentages
for index in "${indices[@]}"; do
    echo "Processing index: $index"

    # lines ~72: rank with BoostIn
    python3 rank_train_samples.py "$boostin_file" "${substring}_ranked" "$index"

    # lines ~75: rank with LCA
    python3 rank_train_samples.py "$lca_file" "${substring}_ranked" "$index"

    # Iterate over each percentage
    for percentage in "${percentages[@]}"; do
        echo "Processing percentage: ${percentage}%"

        # Generate custom data for BoostIn influence file
        python3 generate_custom_data.py "${substring}_ranked_column_${index}_BoostIn.csv" \
                                        "${substring}.train.csv" \
                                        "${percentage}%" \
                                        "${substring}" \
                                        "${task_type}"

        # Generate custom data for LCA influence file
        python3 generate_custom_data.py "${substring}_ranked_column_${index}_LCA.csv" \
                                        "${substring}.train.csv" \
                                        "${percentage}%" \
                                        "${substring}" \
                                        "${task_type}"
    done
    echo "-----------------------------------------------------------------------------------------------------------------"
done

echo "Rankings Complete for indices ${indices[@]}"
echo "Removals Complete for percentages ${percentages[@]}"
echo "Custom data sets have been prepared!"
