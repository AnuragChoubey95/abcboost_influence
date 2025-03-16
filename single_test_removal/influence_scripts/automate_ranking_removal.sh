#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No argument provided. Please provide a substring."
    exit 1
fi

# Get the substring from the command-line argument
substring="$1"

# Determine the number of columns in the influence files
num_columns=$(head -n 1 ../../influence_scores/${substring}.test.csv_mart_J20_v0.1BoostIn_Influence.csv | awk -F',' '{print NF}')

if [ -z "$num_columns" ] || [ "$num_columns" -eq 0 ]; then
    echo "Error: Unable to determine the number of columns in the influence file."
    exit 1
fi

echo "Number of columns in the influence files: $num_columns"

# Generate random unique indices between 0 and (num_columns - 1)
indices=($(python3 -c "import random; print(' '.join(map(str, random.sample(range(0, $num_columns), 100))))"))

# Define percentages
percentages=(0.1 0.5 1 1.5 2)

# Iterate over each index
for index in "${indices[@]}"; do
    echo "Processing index: $index"

    # Rank train samples using BoostIn influence file
    python3 rank_train_samples.py ../../influence_scores/${substring}.test.csv_mart_J20_v0.1BoostIn_Influence.csv ${substring}_ranked $index

    # Rank train samples using LCA influence file
    python3 rank_train_samples.py ../../influence_scores/${substring}.test.csv_mart_J20_v0.1LCA_Influence.csv ${substring}_ranked $index

    # Iterate over each percentage
    for percentage in "${percentages[@]}"; do
        echo "Processing percentage: ${percentage}%"

        # Generate custom data for BoostIn influence file
        python3 generate_custom_data.py ${substring}_ranked_column_${index}_BoostIn.csv ${substring}.train.csv ${percentage}% ${substring}

        # Generate custom data for LCA influence file
        python3 generate_custom_data.py ${substring}_ranked_column_${index}_LCA.csv ${substring}.train.csv ${percentage}% ${substring}
    done
    echo "-----------------------------------------------------------------------------------------------------------------"
done

echo "Rankings Complete for indices ${indices[@]}"
echo "Removals Complete for percentages ${percentages[@]}"
echo "Custom data sets have been prepared!"
