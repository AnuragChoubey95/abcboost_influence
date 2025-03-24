import sys
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset_task_map(filename):
    dataset_task_map = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'\s*\[(.+?)\]=\"(.+?)\"', line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map


def get_task(dataset_substring, dataset_task_map):
    for dataset_name, task in dataset_task_map.items():
        if dataset_substring in dataset_name:
            return dataset_name, task
    return None, None


def split_and_save_data(dataset_substring):
    source_file = os.path.join('..', '..', 'data', f'{dataset_substring}.test.csv')
    target_dir = os.path.join('..', 'custom_data', dataset_substring)

    if not os.path.exists(source_file):
        print(f"Source file '{source_file}' not found.")
        return []

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    data = pd.read_csv(source_file)
    validation_set, held_out_set = train_test_split(data, test_size=0.9, random_state=42)

    validation_set.to_csv(os.path.join(target_dir, f'{dataset_substring}_validate.csv'), index=False)
    held_out_set.to_csv(os.path.join(target_dir, f'{dataset_substring}_held_out.csv'), index=False)

    selected_indices = validation_set.index.tolist()
    return selected_indices


def construct_influence_filename(dataset_substring, method, influence_type):
    p2_substring = "_p2" if method == "regression" else ""
    filename = (
        f"{dataset_substring}.test.csv_{method}_J20_v0.1_p2{influence_type}_Influence.csv"
        if method == "regression"
        else f"{dataset_substring}.test.csv_{method}_J20_v0.1{influence_type}_Influence.csv"
    )
    return filename


def compute_and_save_rankings(dataset_substring, selected_indices, dataset_task_map, influence_type="BoostIn"):
    dataset_name, task = get_task(dataset_substring, dataset_task_map)

    if not dataset_name:
        print(f"No task found for dataset substring '{dataset_substring}'.")
        return

    method = "robustlogit" if task == "binary" else ("mart" if task == "multiclass" else "regression")
    filename = construct_influence_filename(dataset_substring, method, influence_type)
    source_file = os.path.join("..", "..", "influence_scores", filename)

    if not os.path.exists(source_file):
        print(f"Influence file '{source_file}' not found.")
        return

    influence_data = pd.read_csv(source_file)

    selected_columns = influence_data.iloc[:, selected_indices]
    influence_data['row_sum'] = selected_columns.sum(axis=1)
    ranked_indices = influence_data['row_sum'].sort_values(ascending=False).index.tolist()

    target_dir = os.path.join("..", "ranked")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    output_file = os.path.join(target_dir, f"{dataset_substring}_{influence_type}_ranked_10%_test.csv")
    pd.DataFrame({'ranked_indices': ranked_indices}).to_csv(output_file, index=False)

    print(f"Ranked indices saved to '{output_file}'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset_substring>")
        sys.exit(1)

    dataset_substring = sys.argv[1]
    zsh_file = "dataset_task_map.sh"

    dataset_task_map = load_dataset_task_map(zsh_file)

    selected_indices = split_and_save_data(dataset_substring)
    if selected_indices:
        for influence_type in ["BoostIn", "LCA"]:
            compute_and_save_rankings(dataset_substring, selected_indices, dataset_task_map, influence_type)
