import os
import glob
import subprocess
import re

def load_dataset_task_map(filename):
    dataset_task_map = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'\s*\[(.+?)\]=\"(.+?)\"', line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map

def construct_model_filename(train_filename, method, J=20, v=0.1, lp=None, search=None, gap=None):
    base = train_filename
    if method == "regression":
        base += f"_{method}_J{J}_v{v}_p{lp}.model"
    else:
        base += f"_{method}_J{J}_v{v}.model"
    return base

def run_training_and_testing_all(dataset_task_map):
    """
    Iterate over every dataset_substring in dataset_task_map.
    For each one, find CSVs that contain 'validation.csv' in ../mislabelled_train_data,
    then train a model (abcboost_train) and test with abcboost_predict on
    ../validation_test_data/<dataset_substring>/<dataset_substring>_validation.csv
    """
    train_data_dir = os.path.join("..", "mislabelled_train_data")
    if not os.path.exists(train_data_dir):
        print(f"Custom data directory '{train_data_dir}' not found.")
        return

    for dataset_substring, task in dataset_task_map.items():
        print(f"\n=== Processing dataset: {dataset_substring}, task={task} ===")

        # Path to the validation file for this dataset
        test_data_file = os.path.join(
            "..", "validation_test_data", f"{dataset_substring}_validation.csv"
        )
        if not os.path.exists(test_data_file):
            print(f"Validation file '{test_data_file}' not found. Skipping this dataset.")
            continue

        # List CSVs in mislabelled_train_data that might belong to this dataset
        for filename in os.listdir(train_data_dir):
            if not filename.endswith(".csv"):
                continue
            if dataset_substring not in filename:
                continue
           
            train_data_file = os.path.join(train_data_dir, filename)
            print(f"Training data file: {train_data_file}")

            # Build up the abcboost_train command
            if task == "regression":
                method = "regression"
                lp = 2
                train_command = [
                    "../.././abcboost_train", "-method", method, "-lp", str(lp),
                    "-data", train_data_file, "-J", "20", "-v", "0.1", "-iter", "1000"
                ]
            elif task == "binary":
                method = "robustlogit"
                train_command = [
                    "../.././abcboost_train", "-method", method,
                    "-data", train_data_file, "-J", "20", "-v", "0.1", "-iter", "1000"
                ]
            elif task == "multiclass":
                method = "mart"
                search, gap = 2, 10
                train_command = [
                    "../.././abcboost_train", "-method", method,
                    "-data", train_data_file, "-J", "20", "-v", "0.1",
                    "-iter", "1000", "-search", str(search), "-gap", str(gap)
                ]
            else:
                print(f"Unknown task type '{task}' for dataset '{dataset_substring}'. Skipping.")
                continue

            # Construct model filename accurately based on training parameters
            model_name = construct_model_filename(
                train_filename=filename,
                method=method,
                lp=2 if task == "regression" else None,
                search=2 if task == "multiclass" else None,
                gap=10 if task == "multiclass" else None
            )

            # Build up the abcboost_predict command
            predict_command = [
                "../.././abcboost_predict",
                "-data", test_data_file,
                "-model", model_name
            ]

            # Execute training and testing
            print(f"Running training command: {' '.join(train_command)}")
            subprocess.run(train_command)
            print(f"Running testing command: {' '.join(predict_command)}")
            subprocess.run(predict_command)

if __name__ == "__main__":
    zsh_file = "dataset_task_map.sh"
    if not os.path.exists(zsh_file):
        print(f"Task map file '{zsh_file}' not found.")
    else:
        dataset_task_map = load_dataset_task_map(zsh_file)
        run_training_and_testing_all(dataset_task_map)

    # Clean up leftover files
    subprocess.run("rm *.model *.prediction *log", shell=True)