#!/bin/zsh

# Usage: ./run_pipeline.sh <substring>
substring=$1

if [ -z "$substring" ]; then
    echo "Error: No substring argument provided."
    echo "Usage: ./run_pipeline.sh <substring>"
    exit 1
fi

echo "Running automate_ranking_relabel.py with substring: $substring"
zsh automate_ranking_relabel.sh "$substring"

echo "Running train_custom.sh with substring: $substring"
zsh train_custom.sh "$substring"

echo "Running get_average_loss.py with substring: $substring"
zsh get_average_loss.sh "$substring"

echo "Pipeline completed for substring: $substring"
