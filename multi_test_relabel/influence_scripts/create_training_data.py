import sys
import os
import pandas as pd
import numpy as np

# Define your dataset task mapping here
dataset_task_map = {
    "bank_marketing": "binary",
    "htru2": "binary",
    "credit_card": "binary",
    "diabetes": "binary",
    "german": "binary",
    "spambase": "binary",
    "flight_delays": "binary",
    "no_show": "binary",

    "dry_bean": "multiclass",
    "adult": "multiclass",

    "concrete": "regression",
    "energy": "regression",
    "power_plant": "regression",
    "wine_quality": "regression",
    "life_expectancy": "regression"
}


def load_ranked_indices(dataset_substring, influence_type):
    ranked_file = os.path.join("..", "ranked", f"{dataset_substring}_{influence_type}_ranked_10%_test.csv")
    if not os.path.exists(ranked_file):
        print(f"Ranked file '{ranked_file}' not found.")
        return []

    ranked_data = pd.read_csv(ranked_file)
    return ranked_data['ranked_indices'].tolist()


def add_noise_to_labels(data, task, indices, seed=42):
    np.random.seed(seed)
    noisy_data = data.copy()
    if task == "binary":
        for idx in indices:
            noisy_data.iloc[idx, 0] = 1 - noisy_data.iloc[idx, 0]
    elif task == "multiclass":
        classes = noisy_data.iloc[:, 0].unique()
        for idx in indices:
            noisy_data.iloc[idx, 0] = np.random.choice(classes)
    elif task == "regression":
        y_min, y_max = noisy_data.iloc[:, 0].min(), noisy_data.iloc[:, 0].max()
        for idx in indices:
            noisy_data.iloc[idx, 0] = np.random.uniform(y_min, y_max)
    return noisy_data


def create_noisy_train_set(dataset_substring, ranked_indices, influence_type, task):
    train_file = os.path.join("..", "..", "data", f"{dataset_substring}.train.csv")
    target_dir = os.path.join("..", "custom_data", dataset_substring)

    if not os.path.exists(train_file):
        print(f"Train file '{train_file}' not found.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_data = pd.read_csv(train_file)

    for noise_percentage in range(5, 55, 5):
        num_to_noise = int((noise_percentage / 100) * len(ranked_indices))
        indices_to_noise = ranked_indices[:num_to_noise]

        noisy_train_data = add_noise_to_labels(train_data, task, indices_to_noise, seed=42)

        output_file = os.path.join(target_dir, f"{dataset_substring}_{influence_type}_{noise_percentage}percent_noisy_train.csv")
        noisy_train_data.to_csv(output_file, index=False)

        print(f"Dataset with top {noise_percentage}% samples noised saved to '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_custom.py <dataset_substring>")
        sys.exit(1)

    dataset_substring = sys.argv[1]
    task = dataset_task_map.get(dataset_substring)

    if task is None:
        print(f"Task for dataset '{dataset_substring}' not found.")
        sys.exit(1)

    for influence_type in ["BoostIn", "LCA"]:
        ranked_indices = load_ranked_indices(dataset_substring, influence_type)
        if ranked_indices:
            create_noisy_train_set(dataset_substring, ranked_indices, influence_type, task)
