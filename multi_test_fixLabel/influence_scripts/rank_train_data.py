import os
import re
import pandas as pd
import numpy as np

def load_dataset_task_map(file_path="dataset_task_map.sh"):
    """
    Parse the dataset_task_map.sh file returning a dict
    that maps dataset_substring -> task_type.
    Example lines:
      [compas]="binary"
      [dry_bean]="multiclass"
      [concrete]="regression"
    """
    dataset_task_map = {}
    pattern = re.compile(r'\s*\[(.+?)\]=\"(.+?)\"')
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map

def build_influence_filename(dataset_substring, task, method_type):
    """
    Construct the appropriate influence file name.
    - task = 'binary', 'multiclass', or 'regression'
    - method_type = 'BoostIn' or 'LCA'
    Pattern: 
      {dataset_substring}_validation.csv_{method}_J20_v0.1{_p2}{method_type}_Influence.csv
    """
    if task == "binary":
        method_str = "robustlogit"
        # e.g. compas_validation.csv_robustlogit_J20_v0.1BoostIn_Influence.csv
        filename = f"{dataset_substring}_validation.csv_{method_str}_J20_v0.1{method_type}_Influence.csv"
    elif task == "multiclass":
        method_str = "mart"
        # e.g. dry_bean_validation.csv_mart_J20_v0.1BoostIn_Influence.csv
        filename = f"{dataset_substring}_validation.csv_{method_str}_J20_v0.1{method_type}_Influence.csv"
    elif task == "regression":
        method_str = "regression"
        # e.g. concrete_validation.csv_regression_J20_v0.1_p2BoostIn_Influence.csv
        filename = f"{dataset_substring}_validation.csv_{method_str}_J20_v0.1_p2{method_type}_Influence.csv"
    else:
        raise ValueError(f"Unknown task type: {task}")

    return filename

def rank_training_data():
    """
    Main logic:
    For each dataset in dataset_task_map:
      1) For both 'BoostIn' and 'LCA':
         - Build the influence file name accordingly
         - e.g. compas_validation.csv_robustlogit_J20_v0.1BoostIn_Influence.csv
         - Open file in ../../mislabelled_influence_scores
         - Sum across columns => aggregated_influence
         - Sort descending => assigned rank
         - Load original data from ../../data/{dataset}.train.csv
         - Load mislabelled data from ../mislabelled_train_data/{dataset}_mislabel.train.csv
         - Insert rank column, then save to ../ranked_training/{dataset}_{method_type}_{original|mislabel}_ranked.csv
    """
    influence_dir = os.path.join("influence_scores")
    data_dir = os.path.join("..", "..", "data")
    mislabel_dir = os.path.join("..", "mislabelled_train_data")
    output_dir = os.path.join("..", "ranked_training")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset-task map
    task_map = load_dataset_task_map()

    methods = ["BoostIn", "LCA"]

    for dataset_substring, task_type in task_map.items():
        for method_type in methods:
            # Construct the influence file name 
            influence_filename = build_influence_filename(dataset_substring, task_type, method_type)
            influence_file = os.path.join(influence_dir, influence_filename)

            if not os.path.exists(influence_file):
                print(f"[WARN] Influence file not found: {influence_file} for {dataset_substring}, skipping.")
                continue

            print(f"[INFO] Processing influence file => {influence_file}")
            # Influence data shape: (#train_samples, #test_samples)
            influence_df = pd.read_csv(influence_file)

            # Sum across all columns => aggregated influence per row
            influence_df["aggregated_influence"] = influence_df.sum(axis=1)

            # Sort by aggregated_influence descending
            influence_df_sorted = influence_df.sort_values("aggregated_influence", ascending=False)
            influence_df_sorted["rank"] = range(1, len(influence_df_sorted) + 1)

            # We'll create a map from row_index -> rank
            rank_map = dict(zip(influence_df_sorted.index, influence_df_sorted["rank"]))

            # Original training file
            original_train_file = os.path.join(data_dir, f"{dataset_substring}.train.csv")
            if not os.path.exists(original_train_file):
                print(f"[WARN] Original train file missing: {original_train_file}")
                continue

            # Mislabelled training file
            mislabel_file = os.path.join(mislabel_dir, f"{dataset_substring}_mislabel.train.csv")
            if not os.path.exists(mislabel_file):
                print(f"[WARN] Mislabelled file missing: {mislabel_file}")
                continue

            orig_df = pd.read_csv(original_train_file)
            mis_df = pd.read_csv(mislabel_file)

            if len(orig_df) != len(influence_df):
                print(f"[ERROR] Original train rows ({len(orig_df)}) != Influence rows ({len(influence_df)}) for {dataset_substring}. Skipping.")
                continue

            if len(mis_df) != len(influence_df):
                print(f"[ERROR] Mislabelled train rows ({len(mis_df)}) != Influence rows ({len(influence_df)}) for {dataset_substring}. Skipping.")
                continue

            # Add rank column
            orig_df["rank"] = orig_df.index.map(rank_map)
            mis_df["rank"] = mis_df.index.map(rank_map)

            # Save to ../ranked_training
            ranked_orig_file = os.path.join(output_dir, f"{dataset_substring}_{method_type}_original_ranked.csv")
            ranked_mis_file = os.path.join(output_dir, f"{dataset_substring}_{method_type}_mislabel_ranked.csv")

            orig_df.to_csv(ranked_orig_file, index=False)
            mis_df.to_csv(ranked_mis_file, index=False)

            print(f"[INFO] Saved ranked original => {ranked_orig_file}")
            print(f"[INFO] Saved ranked mislabel => {ranked_mis_file}")

if __name__ == "__main__":
    rank_training_data()
