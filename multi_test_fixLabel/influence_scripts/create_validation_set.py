import sys
import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

def load_dataset_task_map(zsh_file="dataset_task_map.sh"):
    """
    Parses the dataset_task_map.sh file and returns a dict
    mapping dataset_substring -> task_type.
    """
    dataset_task_map = {}
    with open(zsh_file, 'r') as file:
        for line in file:
            match = re.match(r'\s*\[(.+?)\]=\"(.+?)\"', line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map

def create_validation_test_data():
    """
    For each dataset substring found in the dataset_task_map,
    this script attempts to load <dataset_substring>.test.csv
    from ../../data. It then splits off 10% of this file (seed=42)
    as a validation test set and saves it into ../validation_test_data.
    The remaining 90% is discarded.
    """
    data_dir = os.path.join('..', '..', 'data')
    output_dir = os.path.join('..', 'validation_test_data')
    os.makedirs(output_dir, exist_ok=True)

    # Load mapping
    task_map = load_dataset_task_map()

    for dataset_substring, task_type in task_map.items():
        source_file = os.path.join(data_dir, f'{dataset_substring}.test.csv')
        if not os.path.exists(source_file):
            print(f"Source file '{source_file}' not found. Skipping.")
            continue

        # Read entire dataset
        data = pd.read_csv(source_file)
        print(f"[{dataset_substring}] Loaded dataset with {len(data)} rows.")

        # Split off 10% as validation set, discarding 90% (train_part)
        # random_state=42 for reproducibility
        train_part, validation_set = train_test_split(data, test_size=0.1, random_state=42)

        print(f"[{dataset_substring}] After split: Train part has {len(train_part)} rows, "
              f"Validation has {len(validation_set)} rows.")
        print(f"[{dataset_substring}] Validation row indices:", validation_set.index.tolist())

        # Save the 10% validation set
        out_file = os.path.join(output_dir, f'{dataset_substring}_validation.csv')
        validation_set.to_csv(out_file, index=False)

        print(f"[{dataset_substring}] Validation set (10%) saved to '{out_file}'.")

if __name__ == "__main__":
    create_validation_test_data()
