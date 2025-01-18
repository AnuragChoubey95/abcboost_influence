import os
import numpy as np
import re
from collections import defaultdict
import time

def preprocess_loss_directory(loss_dir):
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
        if file.endswith(".csv"):
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
            elif file == "comp_cpu.train.csv_regression_J20_v0.1_p2.model_test_sample_losses.csv":
                loss_files["original"].append(file)

    return unique_columns, unique_percentages, loss_files

def compare_losses_and_rank(loss_dir, unique_columns, unique_percentages, loss_files):
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

    # Load the original loss file
    original_loss_file = "comp_cpu.train.csv_regression_J20_v0.1_p2.model_test_sample_losses.csv"
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
        time.sleep(0.5)
        for percentage in unique_percentages:
            print(f"  Processing percentage: {percentage}%")
            time.sleep(0.5)
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
                losses_data = np.loadtxt(os.path.join(loss_dir, file), delimiter=',')
                losses_indices = losses_data[:, 0]
                losses_values = losses_data[:, 1]

                # Find the matching index in the original file
                if column_index in losses_indices:
                    idx_in_original = np.where(original_indices == column_index)[0][0]
                    idx_in_losses = np.where(losses_indices == column_index)[0][0]

                    original_loss = original_losses[idx_in_original]
                    current_loss = losses_values[idx_in_losses]
                    delta_loss = current_loss - original_loss

                    print(f"    Test Index: {column_index}")
                    print(f"      Original Loss: {original_loss:.6f}")
                    print(f"      Current Loss: {current_loss:.6f}")
                    print(f"      Delta Loss: {delta_loss:.6f}")
                    time.sleep(0.5)

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

    print("\nRanking Methods by Average Loss Increase:")
    method_ranking = []
    for percentage in unique_percentages:
        boostin_avg = avg_increase["BoostIn"].get(percentage, 0)
        lca_avg = avg_increase["LCA"].get(percentage, 0)
        print(f"  Percentage {percentage}%:")
        print(f"    BoostIn Average Loss Increase: {boostin_avg:.6f}")
        print(f"    LCA Average Loss Increase: {lca_avg:.6f}")
        if boostin_avg < lca_avg:
            method_ranking.append((percentage, "LCA", lca_avg))
        else:
            method_ranking.append((percentage, "BoostIn", boostin_avg))

    print("\nFinal Rankings by Percentage:")
    for rank in method_ranking:
        print(f"  Percentage {rank[0]}%: Best Method: {rank[1]}, Average Loss Increase: {rank[2]:.6f}")

    return avg_increase, method_ranking

def main():
    # Directory containing loss files
    loss_dir = "../loss_comp/"

    # Preprocess the directory
    unique_columns, unique_percentages, loss_files = preprocess_loss_directory(loss_dir)

    # Print unique columns and percentages
    print(f"Unique Columns: {sorted(unique_columns)}")
    print(f"Unique Percentages: {sorted(unique_percentages)}")
    time.sleep(0.5)

    # Compare losses and rank methods
    avg_increase, method_ranking = compare_losses_and_rank(loss_dir, unique_columns, unique_percentages, loss_files)

if __name__ == "__main__":
    main()
