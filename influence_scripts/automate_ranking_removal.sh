#!/bin/bash

# Generate 10 random unique indices between 0 and 4000
indices=($(python3 -c "import random; print(' '.join(map(str, random.sample(range(0, 4095), 100))))"))

# Define percentages
percentages=(0.1 0.5 1 1.5 2)

# Iterate over each index
for index in "${indices[@]}"; do
    echo "Processing index: $index"

    # Rank train samples using BoostIn influence file
    python3 rank_train_samples.py comp_cpu.test.csv_regression_J20_v0.1_p2BoostIn_Influence.csv comp_cpu_ranked $index

    # Rank train samples using LCA influence file
    python3 rank_train_samples.py comp_cpu.test.csv_regression_J20_v0.1_p2LCA_Influence.csv comp_cpu_ranked $index

    # Iterate over each percentage
    for percentage in "${percentages[@]}"; do
        echo "Processing percentage: ${percentage}%"

        # Generate custom data for BoostIn influence file
        python3 generate_custom_data.py comp_cpu_ranked_column_${index}_BoostIn.csv comp_cpu.train.csv ${percentage}%

        # Generate custom data for LCA influence file
        python3 generate_custom_data.py comp_cpu_ranked_column_${index}_LCA.csv comp_cpu.train.csv ${percentage}%
    done
    echo "-----------------------------------------------------------------------------------------------------------------"
done

echo "Rankings Complete for indices ${indices[@]}"
echo "Removals Complete for percentages ${percentages[@]}"
echo "Custom data sets have been prepared!"