import sys
import os
import re
import pandas as pd


def load_dataset_task_map(filename):
    dataset_task_map = {}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'\s*\[(.+?)\]=\"(.+?)\"', line)
            if match:
                dataset, task = match.groups()
                dataset_task_map[dataset] = task
    return dataset_task_map


def construct_loss_filename(dataset_substring, method, percentage=None, influence_type=None, task_type=None):
    p2_substring = "_p2" if task_type == "regression" else ""

    if percentage and influence_type:
        filename = f"{dataset_substring}_{influence_type}_{percentage}percent_noisy_train.csv_{method}_J20_v0.1{p2_substring}.model_test_sample_losses.csv"
    else:
        filename = f"{dataset_substring}.train.csv_{method}_J20_v0.1{p2_substring}.model_test_sample_losses.csv"

    return filename


def calculate_loss_difference(original_file, comparison_file):
    orig_data = pd.read_csv(original_file)
    comp_data = pd.read_csv(comparison_file)

    orig_loss = orig_data.values.mean()
    comp_loss = comp_data.values.mean()

    delta = comp_loss - orig_loss

    return delta


def write_statistics(dataset_substring, dataset_task_map):
    task = dataset_task_map.get(dataset_substring)

    if not task:
        print(f"Task not found for dataset '{dataset_substring}'.")
        return

    method = "robustlogit" if task == "binary" else ("mart" if task == "multiclass" else "regression")

    original_loss_file = os.path.join("loss_comp", construct_loss_filename(dataset_substring, method, task_type=task))

    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append(f"Dataset: {dataset_substring}")
    output_lines.append("=" * 60)
    
    percentage_data = {}

    for percentage in range(5, 55, 5):
        results = {}
        for influence_type in ["BoostIn", "LCA"]:
            comp_loss_file = os.path.join("loss_comp", construct_loss_filename(dataset_substring, method, percentage, influence_type, task))
            
            if os.path.exists(original_loss_file) and os.path.exists(comp_loss_file):
                delta_loss = calculate_loss_difference(original_loss_file, comp_loss_file)
                results[influence_type] = delta_loss
            else:
                missing_files = []
                if not os.path.exists(original_loss_file):
                    missing_files.append("Original file missing")
                if not os.path.exists(comp_loss_file):
                    missing_files.append(f"{influence_type} file missing")
                results[influence_type] = " / ".join(missing_files)

        percentage_data[percentage] = results

    with open("../statistics.txt", "a") as stats_file:
        for percentage, results in percentage_data.items():
            output_lines.append(f"\n--- {percentage}% Relabel ---")
            if isinstance(results["BoostIn"], float) and isinstance(results["LCA"], float):
                greater_method = "BoostIn" if results["BoostIn"] > results["LCA"] else "LCA"
                output_lines.append(f"  BoostIn: {results['BoostIn']:.6f}")
                output_lines.append(f"  LCA: {results['LCA']:.6f}")
                output_lines.append(f"  Greater Delta: {greater_method}")
            else:
                output_lines.append(f"  BoostIn: {results['BoostIn']}")
                output_lines.append(f"  LCA: {results['LCA']}")
        
        stats_file.write("\n".join(output_lines) + "\n")

    print("\n".join(output_lines))  # Print it for immediate feedback



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_change_loss.py <dataset_substring>")
        sys.exit(1)

    dataset_substring = sys.argv[1]
    dataset_task_map = load_dataset_task_map("dataset_task_map.sh")

    write_statistics(dataset_substring, dataset_task_map)