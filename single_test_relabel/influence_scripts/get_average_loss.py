import os
import re
import argparse
import numpy as np
import time
from collections import defaultdict

def preprocess_loss_directory(loss_dir, substring, task_type):
    """
    Scans all CSV files in 'loss_dir' that match 'substring', detecting:
      - unique columns (by regex `_column(\d+)`)
      - unique percentages (by regex `filtered_(\d+(\.\d+)?)%`)
      - method categorization (BoostIn/LCA)
      - which file is the 'original' file, based on task_type.
    
    Returns:
      unique_columns (set of int)
      unique_percentages (set of float)
      loss_files (dict of lists):
         {
           "original": [...],
           "BoostIn": [...],
           "LCA": [...]
         }
    """
    unique_columns = set()
    unique_percentages = set()  # <--- now starting as empty, discovered from filenames
    loss_files = defaultdict(list)

    for file in os.listdir(loss_dir):
        # We only consider CSV files containing the user-specified 'substring'
        if file.endswith(".csv") and substring in file:
            # 1) Identify the original file (task_type-based)
            if task_type == "regression":
                if file == f"{substring}.train.csv_regression_J20_v0.1_p2.model_test_sample_losses.csv":
                    loss_files["original"].append(file)
            elif task_type == "multiclass":
                if file == f"{substring}.train.csv_mart_J20_v0.1.model_test_sample_losses.csv":
                    loss_files["original"].append(file)
            else:  # default 'binary'
                if file == f"{substring}.train.csv_robustlogit_J20_v0.1.model_test_sample_losses.csv":
                    loss_files["original"].append(file)

            # 2) Attempt to detect "columnXYZ" in the filename
            column_match = re.search(r"_column(\d+)", file)
            if column_match:
                col_idx = int(column_match.group(1))
                unique_columns.add(col_idx)

            # 3) Attempt to detect "filtered_XX%" in the filename
            percentage_match = re.search(r"relabelled_(\d+(?:\.\d+)?)%", file)
            if percentage_match:
                val = float(percentage_match.group(1))
                unique_percentages.add(val)

            # 4) Classify method
            if "BoostIn" in file:
                loss_files["BoostIn"].append(file)
            elif "LCA" in file:
                loss_files["LCA"].append(file)
    
    return unique_columns, unique_percentages, loss_files


def compare_losses_and_rank(loss_dir, unique_columns, unique_percentages,
                            loss_files, substring, task_type):
    """
    Loads the 'original' loss file (task_type determines naming),
    and then compares each (column, percentage) variant of BoostIn/LCA
    to compute average deltas, etc.
    """

    comparison_results = {
        "BoostIn": defaultdict(list),
        "LCA": defaultdict(list)
    }

    # Figure out the original file name from task_type
    if task_type == "regression":
        original_loss_file = f"{substring}.train.csv_regression_J20_v0.1_p2.model_test_sample_losses.csv"
    elif task_type == "multiclass":
        original_loss_file = f"{substring}.train.csv_mart_J20_v0.1.model_test_sample_losses.csv"
    else:
        original_loss_file = f"{substring}.train.csv_robustlogit_J20_v0.1.model_test_sample_losses.csv"

    original_path = os.path.join(loss_dir, original_loss_file)
    print(f"Loading original loss file: {original_loss_file}")
    
    time.sleep(0.5)
    original_data = np.loadtxt(os.path.join(loss_dir, original_loss_file), delimiter=',')
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
                percentage_str = f"relabelled_{percentage:.1f}%".rstrip(".0%")
                if percentage_str not in file:
                    continue

                # Extract column index from filename
                column_match = re.search(r"column(\d+)", file)
                if not column_match:
                    continue
                column_index = int(column_match.group(1))

                # Load the loss file
                losses_data = np.loadtxt(os.path.join(loss_dir, file), delimiter=',')
                losses_indices = losses_data[:, 0]
                losses_values = losses_data[:, 1]

                # Find the matching index in the original file
                if column_index in losses_indices:
                    if column_index not in original_indices:
                        print(f"Error: Column index {column_index} not found in original indices.")
                        # continue
                    idx_in_original = np.where(original_indices == column_index)[0][0]
                    idx_in_losses = np.where(losses_indices == column_index)[0][0]

                    original_loss = original_losses[idx_in_original]
                    current_loss = losses_values[idx_in_losses]
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
            if boostin_avg < lca_avg:
                method_ranking.append((percentage, "LCA", lca_avg))
            else:
                method_ranking.append((percentage, "BoostIn", boostin_avg))
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
    parser.add_argument("task_type", type=str, help="Task type (binary, multiclass, or regression).")
    args = parser.parse_args()

    substring = args.substring
    task_type = args.task_type
    print(task_type)
    # Directory containing your CSV loss files
    loss_dir = os.path.join(os.path.dirname(__file__), "../loss_comp/")

    # 1) Collect the columns, percentages, etc.
    unique_columns, unique_percentages, loss_files = preprocess_loss_directory(
        loss_dir, substring, task_type
    )

    # 2) Print them out to verify
    print(f"Unique Columns: {sorted(unique_columns)}")
    print(f"Unique Percentages: {sorted(unique_percentages)}")
    time.sleep(0.5)

    # 3) Perform the actual comparisons
    compare_losses_and_rank(loss_dir, unique_columns, unique_percentages,
                            loss_files, substring, task_type)


if __name__ == "__main__":
    main()
