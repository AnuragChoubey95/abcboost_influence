#!/usr/bin/env zsh

# This script automatically ranks training samples for a given dataset substring
# and removes top-k influential samples at various percentages.

# Usage: ./automate_ranking_relabel.sh <substring>

# 1) Check arguments
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring."
    exit 1
fi

# 2) Get the substring from command-line
substring="$1"

# 3) Declare dataset -> task map directly (Zsh style)
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

# 4) Lookup task type
task_type="${dataset_task[$substring]}"
if [ -z "$task_type" ]; then
    echo "Warning: No task type found for '$substring'. Defaulting to binary."
    task_type="binary"
fi
echo "Detected task type for '${substring}' => ${task_type}"

# 5) Determine the influence filenames based on task type
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

# 6) Verify that BoostIn file exists
if [ ! -f "$boostin_file" ]; then
    echo "Error: $boostin_file not found. Check your naming or paths."
    exit 1
fi

# 7) Determine number of columns
num_columns=$(head -n 1 "$boostin_file" | awk -F',' '{print NF}')
if [ -z "$num_columns" ] || [ "$num_columns" -eq 0 ]; then
    echo "Error: Unable to determine number of columns in $boostin_file."
    exit 1
fi
echo "Number of columns in the influence files: $num_columns"

# 8) Generate random indices (seed=42)
indices=($(python3 -c "import random; random.seed(42); print(' '.join(map(str, random.sample(range(0, $num_columns), 100))))"))

# 9) Define percentages
percentages=(0.1 0.5 1 1.5 2)

# 10) Rank + Relabel
for index in "${indices[@]}"; do
    echo "Processing index: $index"

    test_input_file="../../data/${substring}.test.csv"
    test_output_dir="../test_data/${substring}"
    test_output_file="${test_output_dir}/${substring}_test_column_${index}.csv"

    mkdir -p "$test_output_dir"

    test_line=$(sed -n "$((index + 1))p" "$test_input_file")

    if [ -z "$test_line" ]; then
        echo "Warning: Could not extract row $index from $test_input_file"
    else
        echo "$test_line" > "$test_output_file"
        echo "Saved test row $index to $test_output_file"
    fi


    python3 rank_train_samples.py "$boostin_file" "${substring}_ranked" "$index"

    python3 rank_train_samples.py "$lca_file" "${substring}_ranked" "$index"

    for percentage in "${percentages[@]}"; do
        echo "Processing percentage: ${percentage}%"

        # Generate custom data - BoostIn
        python3 generate_custom_data.py "${substring}_ranked_column_${index}_BoostIn.csv" \
                                        "${substring}.train.csv" \
                                        "${percentage}%" \
                                        "${substring}" \
                                        "${task_type}"

        # Generate custom data - LCA
        python3 generate_custom_data.py "${substring}_ranked_column_${index}_LCA.csv" \
                                        "${substring}.train.csv" \
                                        "${percentage}%" \
                                        "${substring}" \
                                        "${task_type}"
    done
    echo "-----------------------------------------------------------------------------------------------------------------"
done

echo "Rankings Complete for indices ${indices[@]}"
echo "Relabels Complete for percentages ${percentages[@]}"
echo "Custom data sets have been prepared!"
