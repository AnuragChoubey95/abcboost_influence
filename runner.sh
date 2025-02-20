#!/bin/bash

# Define dataset keywords
# "adult" 
# "htru2"
# "credit_card"
# "flight_delays"
# "diabetes"
# "no_show"
# "german_credit"
# "spambase"
# "surgical"
# "vaccine"
datasets=(  )

# Log file
log_file="Single_Test_Removal.log"
echo "Single_Test_Removal Run Log - $(date)" > "$log_file"

# Function to get current timestamp
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# Loop through each dataset keyword
for dataset in "${datasets[@]}"; do
    echo -e "\nProcessing dataset: $dataset"
    echo "[$(timestamp)] Starting $dataset" >> "$log_file"

    # Start time
    start_time=$(date +%s)

    # Step 1: Run automate_ranking_removal.sh inside single_test_removal/influence_scripts/
    echo "Running automate_ranking_removal.sh for $dataset..."
    cd single_test_removal/influence_scripts/ || { echo "Error: Failed to enter directory single_test_removal/influence_scripts/" | tee -a "$log_file"; exit 1; }
    zsh automate_ranking_removal.sh "$dataset"
    if [ $? -ne 0 ]; then
        echo "Error during automate_ranking_removal.sh for $dataset. Skipping..." | tee -a "$log_file"
        cd ..
        continue
    fi
    cd ..

    # Step 2: Run train_custom.sh with the dataset as argument
    echo "Running train_custom.sh for $dataset..."
    # cd single_test_removal || { echo "Error: Failed to enter directory single_test_removal" | tee -a "$log_file"; exit 1; }
    zsh train_custom.sh "$dataset"
    if [ $? -ne 0 ]; then
        echo "Error during train_custom.sh for $dataset. Skipping..." | tee -a "$log_file"
        cd ..
        continue
    fi
    cd ..  # Navigate back to root directory

    # Step 3: Run get_average_loss.py inside influence_scripts/
    echo "Running get_average_loss.py for $dataset..."
    cd single_test_removal/influence_scripts/ || { echo "Error: Failed to enter directory influence_scripts" | tee -a "$log_file"; exit 1; }
    python3 get_average_loss.py "$dataset"
    if [ $? -ne 0 ]; then
        echo "Error during get_average_loss.py for $dataset. Skipping..." | tee -a "$log_file"
        cd ..
        continue
    fi
    cd ..  # Navigate back to root directory

    # End time
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    formatted_runtime=$(printf "%02d:%02d:%02d" $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))

    # Log success
    echo "[$(timestamp)] Successfully completed $dataset in $formatted_runtime" >> "$log_file"

    # Prompt user to continue
    read -p "Press Enter to continue to the next dataset or Ctrl+C to exit..."
done

echo -e "\nABCBoost Run Completed. Log saved in $log_file"
