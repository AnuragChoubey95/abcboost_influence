import os
import re
import pandas as pd
import numpy as np

np.random.seed(42)


def load_dataset_task_map():
    dataset_task_map = {}
    with open("dataset_task_map.sh", 'r') as file:
        for line in file:
            match = re.match(r'\s*\[(.+?)\]="(.+?)"', line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map


def flip_label(label, task, possible_labels=None, y_min=None, y_max=None):
    if task == "binary":
        return 1 - label
    elif task == "multiclass":
        return np.random.choice(possible_labels)
    elif task == "regression":
        return np.random.uniform(y_min, y_max)
    else:
        return label  


def create_mislabelled_data():
    task_map = load_dataset_task_map()
    data_dir = os.path.join("..", "..", "data")
    output_dir = os.path.join("..", "mislabelled_train_data")
    os.makedirs(output_dir, exist_ok=True)

    for dataset, task in task_map.items():
        input_file = os.path.join(data_dir, f"{dataset}.train.csv")
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        df = pd.read_csv(input_file)
        total = len(df)
        indices_to_flip = np.random.choice(total, int(0.4 * total), replace=False)

        if task == "multiclass":
            possible_labels = df.iloc[:, 0].unique()
        elif task == "regression":
            y_min, y_max = df.iloc[:, 0].min(), df.iloc[:, 0].max()
        else:
            possible_labels, y_min, y_max = None, None, None

        for idx in indices_to_flip:
            original = df.iloc[idx, 0]
            df.iat[idx, 0] = flip_label(original, task, possible_labels, y_min, y_max)

        output_file = os.path.join(output_dir, f"{dataset}_mislabel.train.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    create_mislabelled_data()
