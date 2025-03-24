import sys
import os
import pandas as pd

def load_ranked_indices(dataset_substring, influence_type):
    ranked_file = os.path.join("..", "ranked", f"{dataset_substring}_{influence_type}_ranked_10%_test.csv")
    if not os.path.exists(ranked_file):
        print(f"Ranked file '{ranked_file}' not found.")
        return []

    ranked_data = pd.read_csv(ranked_file)
    return ranked_data['ranked_indices'].tolist()


def create_ranked_train_set(dataset_substring, ranked_indices, influence_type):
    train_file = os.path.join("..", "..", "data", f"{dataset_substring}.train.csv")
    target_dir = os.path.join("..", "custom_data", dataset_substring)

    if not os.path.exists(train_file):
        print(f"Train file '{train_file}' not found.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_data = pd.read_csv(train_file)

    total_rows = len(train_data)

    for removal_percentage in range(5, 55, 5):
        num_to_remove = int((removal_percentage / 100) * len(ranked_indices))
        indices_to_remove = set(ranked_indices[:num_to_remove])

        remaining_indices = [idx for idx in range(total_rows) if idx not in indices_to_remove]

        reduced_train_data = train_data.iloc[remaining_indices]

        output_file = os.path.join(target_dir, f"{dataset_substring}_{influence_type}_{removal_percentage}percent_removed_train.csv")
        reduced_train_data.to_csv(output_file, index=False)

        print(f"Dataset with top {removal_percentage}% samples removed saved to '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_custom.py <dataset_substring>")
        sys.exit(1)

    dataset_substring = sys.argv[1]

    for influence_type in ["BoostIn", "LCA"]:
        ranked_indices = load_ranked_indices(dataset_substring, influence_type)
        if ranked_indices:
            create_ranked_train_set(dataset_substring, ranked_indices, influence_type)