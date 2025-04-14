import sys
import os
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


def construct_model_filename(train_filename, method, J=20, v=0.1, lp=None):
    base = train_filename
    if method == "regression":
        base += f"_{method}_J{J}_v{v}_p{lp}.model"
    else:
        base += f"_{method}_J{J}_v{v}.model"
    return base


def run_initial_baseline(dataset_substring, task):
    print("\n--- Running Baseline Training and Evaluation ---")

    train_data_file = os.path.join("..", "..", "data", f"{dataset_substring}.train.csv")
    test_data_file = os.path.join("..", "custom_data", dataset_substring, f"{dataset_substring}_held_out.csv")

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
        train_command = [
            "../.././abcboost_train", "-method", method,
            "-data", train_data_file, "-J", "20", "-v", "0.1",
            "-iter", "1000", "-search", "2", "-gap", "10"
        ]
    else:
        print(f"Unknown task type '{task}' for dataset '{dataset_substring}'.")
        return

    model_name = construct_model_filename(
        train_filename=os.path.basename(train_data_file),
        method=method,
        lp=2 if task == "regression" else None
    )

    predict_command = [
        "../.././abcboost_predict",
        "-data", test_data_file,
        "-model", model_name
    ]

    print(f"Running baseline training command: {' '.join(train_command)}")
    subprocess.run(train_command)

    print(f"Running baseline testing command: {' '.join(predict_command)}")
    subprocess.run(predict_command)


def run_training_and_testing(dataset_substring, dataset_task_map):
    task = dataset_task_map.get(dataset_substring)

    if not task:
        print(f"Task not found for dataset '{dataset_substring}'.")
        return

    run_initial_baseline(dataset_substring, task)

    custom_data_dir = os.path.join("..", "custom_data", dataset_substring)
    test_data_file = os.path.join(custom_data_dir, f"{dataset_substring}_held_out.csv")

    if not os.path.exists(custom_data_dir):
        print(f"Custom data directory '{custom_data_dir}' not found.")
        return

    for filename in os.listdir(custom_data_dir):
        if filename.endswith("percent_removed_train.csv"):
            train_data_file = os.path.join(custom_data_dir, filename)

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
                train_command = [
                    "../.././abcboost_train", "-method", method,
                    "-data", train_data_file, "-J", "20", "-v", "0.1",
                    "-iter", "1000", "-search", "2", "-gap", "10"
                ]
            else:
                print(f"Unknown task type '{task}' for dataset '{dataset_substring}'.")
                continue

            model_name = construct_model_filename(
                train_filename=filename,
                method=method,
                lp=2 if task == "regression" else None
            )

            predict_command = [
                "../.././abcboost_predict",
                "-data", test_data_file,
                "-model", model_name
            ]

            print(f"\n--- Running Training with data removal ({filename}) ---")
            print(f"Training command: {' '.join(train_command)}")
            subprocess.run(train_command)

            print(f"Testing command: {' '.join(predict_command)}")
            subprocess.run(predict_command)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_train_test.py <dataset_substring>")
        sys.exit(1)

    dataset_substring = sys.argv[1]
    dataset_task_map = load_dataset_task_map("dataset_task_map.sh")

    run_training_and_testing(dataset_substring, dataset_task_map)