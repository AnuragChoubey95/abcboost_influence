import os
import numpy as np
import re
from collections import defaultdict
import time
import argparse

def parse_dataset_task_map(sh_file_path):
    """
    Parses the dataset_task_map.sh file and returns a Python dictionary.
    """
    dataset_task = {}
    with open(sh_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("declare"):
                continue
            match = re.match(r'\[(.+?)\]="(.+?)"', line)
            if match:
                key, value = match.groups()
                dataset_task[key] = value
    return dataset_task


def construct_loss_filename(substring, task_type):
    """
    Constructs the expected original loss filename for a given dataset and task type.
    """
    if task_type == "binary":
        return f"{substring}.train.csv_robustlogit_J20_v0.1.model_test_sample_losses.csv"
    elif task_type == "multiclass":
        return f"{substring}.train.csv_mart_J20_v0.1.model_test_sample_losses.csv"
    elif task_type == "regression":
        return f"{substring}.train.csv_regression_J20_v0.1_p2.model_test_sample_losses.csv"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")



def preprocess_loss_directory(loss_dir, substring):
    """
    Preprocess the directory to extract unique columns and percentages.

    Args:
        loss_dir (str): Path to the directory containing loss files.

    Returns:
        tuple: A tuple containing:
            - unique_columns (set): Unique test sample indices (columns) in filenames.
            - unique_percentages (set): Unique percentages in filenames.
            - loss_files (dict): Dictionary categorizing files by method (BoostIn or LCA).
    """
    unique_columns = set()
    unique_percentages = set()
    loss_files = defaultdict(list)

    # Extract columns, percentages, and categorize files
    for file in os.listdir(loss_dir):
        if file.endswith(".csv") and substring in file:
            # Extract column and percentage from filename
            column_match = re.search(r"column(\d+)", file)
            percentage_match = re.search(r"filtered_(\d+(\.\d+)?)%", file)

            if column_match:
                unique_columns.add(int(column_match.group(1)))
            if percentage_match:
                unique_percentages.add(float(percentage_match.group(1)))

            # Categorize files by method
            if "BoostIn" in file:
                loss_files["BoostIn"].append(file)
            elif "LCA" in file:
                loss_files["LCA"].append(file)
            elif file == f"{substring}.train.csv_mart_J20_v0.1.model_test_sample_losses.csv":
                loss_files["original"].append(file)

    return unique_columns, unique_percentages, loss_files


def compare_losses_and_rank(original_loss_dir, influence_loss_dir, unique_columns, unique_percentages, loss_files, substring):
    """
    Compare losses for BoostIn and LCA methods with the original loss file and rank them.

    Args:
        loss_dir (str): Path to the directory containing loss files.
        unique_columns (set): Unique test sample indices to consider.
        unique_percentages (set): Unique percentages in filenames.
        loss_files (dict): Dictionary categorizing files by method (BoostIn, LCA, original).

    Returns:
        dict: Average rank and average delta loss for BoostIn and LCA methods.
    """
    comparison_results = {
        "BoostIn": defaultdict(list),
        "LCA": defaultdict(list)
    }

    dataset_task_map = parse_dataset_task_map("dataset_task_map.sh")

    # Load the original loss file
    original_loss_file = construct_loss_filename(substring, dataset_task_map[substring])
    print(f"Loading original loss file: {original_loss_file}")
    time.sleep(0.5)
    original_data = np.loadtxt(os.path.join(original_loss_dir, original_loss_file), delimiter=',')
    original_indices = original_data[:, 0]
    original_losses = original_data[:, 1]

    # Aggregate ranking data
    rankings = {
        "BoostIn": defaultdict(list),
        "LCA": defaultdict(list)
    }

    # Compare losses for each method and each percentage
    for method in ["BoostIn", "LCA"]:
        print(f"\nProcessing method: {method}")
        # time.sleep(0.5)
        for percentage in unique_percentages:
            print(f"  Processing percentage: {percentage}%")
            # time.sleep(0.5)
            for file in loss_files[method]:
                # Ensure the file corresponds to the percentage
                percentage_str = f"filtered_{percentage:.1f}%".rstrip(".0%")
                if percentage_str not in file:
                    continue

                # Extract column index from filename
                column_match = re.search(r"column(\d+)", file)
                if not column_match:
                    continue
                column_index = int(column_match.group(1))

                # Load the loss file
                losses_data = np.loadtxt(os.path.join(influence_loss_dir, file), delimiter=',')

                # losses_data is 1D: [index, loss]
                if losses_data.ndim != 1 or len(losses_data) != 2:
                    print(f"Unexpected format in {file}: {losses_data}")
                    continue

                _ , current_loss = losses_data

                # Find the matching index in the original file
                if column_index in original_indices:
                    idx_in_original = np.where(original_indices == column_index)[0][0]
                    original_loss = original_losses[idx_in_original]
                    delta_loss = current_loss - original_loss
                    print(f"    Test Index: {column_index}")
                    print(f"      Original Loss: {original_loss:.15f}")
                    print(f"      Current Loss: {current_loss:.15f}")
                    print(f"      Delta Loss: {delta_loss:.15f}")
                    # time.sleep(0.5)

                    # Store the results
                    comparison_results[method][percentage].append(delta_loss)

    # Compute average increase in loss and rank the methods
    avg_increase = {
        "BoostIn": {},
        "LCA": {}
    }
    for method in ["BoostIn", "LCA"]:
        for percentage, deltas in comparison_results[method].items():
            avg_delta = np.mean(deltas) if deltas else 0
            avg_increase[method][percentage] = avg_delta

    with open("../statistics.txt", "a") as f:  # Open the file in append mode
        f.write(f"\nSubstring: {substring} || Sample Size of Test Indices: {len(unique_columns)}\n")
        print("\nRanking Methods by Average Loss Increase:")
        f.write("\nRanking Methods by Average Loss Increase:\n")
        method_ranking = []
        for percentage in unique_percentages:
            boostin_avg = avg_increase["BoostIn"].get(percentage, 0)
            lca_avg = avg_increase["LCA"].get(percentage, 0)
            print(f"  Percentage {percentage}%:")
            print(f"    BoostIn Average Loss Increase: {boostin_avg:.15f}")
            print(f"    LCA Average Loss Increase: {lca_avg:.15f}")
            f.write(f"\n  Percentage {percentage}%:")
            f.write(f"\n    BoostIn Average Loss Increase: {boostin_avg:.15f}")
            f.write(f"\n    LCA Average Loss Increase: {lca_avg:.15f}")
            if boostin_avg > lca_avg:
                method_ranking.append((percentage, "BoostIn", boostin_avg))
            else:
                method_ranking.append((percentage, "LCA", lca_avg))

        f.write("\n-----------------------")
        print("\nFinal Rankings by Percentage:")
        f.write("\nFinal Rankings by Percentage:\n")
        f.write(f"\nSubstring: {substring} || Sample Size of Test Indices: {len(unique_columns)}\n")  # Write the substring
        for rank in method_ranking:
            ranking_line = f"  Percentage {rank[0]}%: Best Method: {rank[1]}, Average Loss Increase: {rank[2]:.15f}"
            print(ranking_line)  # Print to console
            f.write(ranking_line + "\n")  # Write to file
        f.write("\n--------------------------------------------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Process loss files based on a substring.")
    parser.add_argument("substring", type=str, help="Substring to filter filenames.")
    args = parser.parse_args()
    substring = args.substring

    # Directory containing loss files
    original_loss_dir = os.path.join(os.path.dirname(__file__), "../../loss_comp/")
    influence_loss_dir = os.path.join(os.path.dirname(__file__), "loss_comp/")
    # Preprocess the directory
    unique_columns, unique_percentages, loss_files = preprocess_loss_directory(influence_loss_dir, substring)

    # Print unique columns and percentages
    print(f"Unique Columns: {sorted(unique_columns)}")
    print(f"Unique Percentages: {sorted(unique_percentages)}")
    time.sleep(0.5)

    # Compare losses and rank methods
    compare_losses_and_rank(original_loss_dir, influence_loss_dir, unique_columns, unique_percentages, loss_files, substring)


if __name__ == "__main__":
    main()
