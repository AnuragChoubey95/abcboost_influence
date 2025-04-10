#!/bin/zsh

substring=$1

if [ -z "$substring" ]; then
    echo "Error: No substring argument provided."
    echo "Usage: ./run_pipeline.sh <substring>"
    exit 1
fi

echo "Running automate_ranking_removal.py with substring: $substring"
python3 automate_ranking_removal.py "$substring"

echo "Running train_custom.sh with substring: $substring"
bash train_custom.sh "$substring"

echo "Running get_average_loss.py with substring: $substring"
python3 get_average_loss.py "$substring"

echo "Pipeline completed for substring: $substring"
