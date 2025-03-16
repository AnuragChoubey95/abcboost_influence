import numpy as np
import argparse
import os

# Function to relabel data based on ranking indices
def relabel_data(ranking_file, data_file, percentage, output_dir, task_type):
    """
    Relabels the output values of top-ranked samples from a dataset based on a ranking file and task type.

    Args:
        ranking_file (str): Path to the ranking output file.
        data_file (str): Path to the original data CSV file.
        percentage (str): Percentage of top-ranked samples to relabel.
        output_dir (str): Directory to save the modified dataset.
        task_type (str): Type of task - 'binary', 'multiclass', or 'regression'.

    Returns:
        None: Writes the modified dataset to the specified output directory.
    """
    # Ensure percentage is a valid format (e.g., "1%", "2%").
    if not percentage.endswith('%'):
        raise ValueError("Percentage must end with '%' (e.g., '1%').")
    try:
        percentage_value = float(percentage.strip('%'))
    except ValueError:
        raise ValueError("Invalid percentage format. Use a numeric value followed by '%'.")

    # Load the ranking file
    ranking_data = np.genfromtxt(ranking_file, delimiter=',', skip_header=1, dtype=int, usecols=0)
    
    # Determine the number of rows to relabel
    num_to_relabel = int((percentage_value / 100) * len(ranking_data))
    print(f"Total ranked samples: {len(ranking_data)}")
    print(f"Relabeling top {num_to_relabel} samples ({percentage}).")

    # Get the indices of the top samples to relabel
    indices_to_relabel = set(ranking_data[:num_to_relabel])

    # Load the data file
    data = np.genfromtxt(data_file, delimiter=',', skip_header=0)
    
    # Identify the column index of the target variable (first column, index 0)
    target_index = 0
    
    # Compute necessary statistics for regression relabeling
    target_values = data[:, target_index]
    y_mean = np.mean(target_values)
    
    # Relabel the selected data points based on the task type
    for i in indices_to_relabel:
        if task_type == 'binary':
            data[i, target_index] = 1 - data[i, target_index]  # Flip the label
        elif task_type == 'multiclass':
            unique_classes = np.unique(target_values)
            possible_classes = unique_classes[unique_classes != data[i, target_index]]
            data[i, target_index] = np.random.choice(possible_classes)  # Randomly assign a different class
        elif task_type == 'regression':
            if data[i, target_index] > y_mean:
                data[i, target_index] = y_mean - (y_mean / 2)
            else:
                data[i, target_index] = y_mean + (y_mean / 2)
        else:
            raise ValueError("Invalid task type. Choose from 'binary', 'multiclass', or 'regression'.")
    
    # Generate the output file path
    os.makedirs(output_dir, exist_ok=True)
    if "LCA" in ranking_file:
        suffix = "_LCA"
    else:
        suffix = "_BoostIn"
    column_index = ranking_file.split('_column_')[1].split('_')[0]
    output_file = os.path.join(output_dir, os.path.basename(data_file).replace('.csv', f'_relabelled_{percentage}_column{column_index}{suffix}.csv'))
    
    # Save the modified dataset
    np.savetxt(output_file, data, delimiter=',', fmt='%s')
    print(f"Relabeled data saved to: {output_file}")

# Main function to handle argument parsing and execution
def main():
    """
    Main function to parse arguments and execute the relabeling process.
    """
    parser = argparse.ArgumentParser(description="Relabel top-ranked samples from a dataset.")
    parser.add_argument("ranking_file", type=str, help="Path to the ranking output file.")
    parser.add_argument("data_file", type=str, help="Path to the dataset CSV file (in ../data/ directory).")
    parser.add_argument("percentage", type=str, help="Percentage of top-ranked samples to relabel (e.g., '1%').")
    parser.add_argument("output_dirID", type=str, help="Unique name of output dir")
    parser.add_argument("task_type", type=str, choices=['binary', 'multiclass', 'regression'], help="Task type: binary, multiclass, or regression.")
    
    args = parser.parse_args()

    # Call the relabel_data function with parsed arguments
    rank_file_path = f"../../influence_scores/{args.ranking_file}"
    data_file_path = f"../../data/{args.data_file}"
    output_path = f"../custom_data/{args.output_dirID}/"
    relabel_data(rank_file_path, data_file_path, args.percentage, output_path, args.task_type)

if __name__ == "__main__":
    main()