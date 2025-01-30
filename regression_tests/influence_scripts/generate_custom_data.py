import numpy as np
import argparse
import os

# Function to filter data based on ranked indices
def filter_data(ranking_file, data_file, percentage, output_dir):
    """
    Filters out the top-ranked samples from a dataset based on a ranking file.

    Args:
        ranking_file (str): Path to the ranking output file (from the previous script).
        data_file (str): Path to the original data CSV file (in ../data/).
        percentage (str): Percentage of top-ranked samples to remove (e.g., "1%", "2%").
        output_dir (str): Directory to save the modified dataset (e.g., ../custom_data/).
        column_index (int): Index of the column used for ranking (to include in output file name).

    Returns:
        None: Writes the modified dataset to the specified output directory.
    """
    # Ensure percentage is a valid format (e.g., "1%", "2%").
    if not percentage.endswith('%'):
        raise ValueError("Percentage must end with '%' (e.g., '1%', '2%').")
    try:
        percentage_value = float(percentage.strip('%'))
    except ValueError:
        raise ValueError("Invalid percentage format. Use a numeric value followed by '%'.")

    # Load the ranking file
    ranking_data = np.genfromtxt(ranking_file, delimiter=',', skip_header=1, dtype=int, usecols=0)
    
    # Determine the number of rows to remove
    num_to_remove = int((percentage_value / 100) * len(ranking_data))
    print(f"Total ranked samples: {len(ranking_data)}")
    print(f"Removing top {num_to_remove} samples ({percentage}).")

    # Get the indices of the top samples to remove
    indices_to_remove = set(ranking_data[:num_to_remove])

    # Load the data file
    data = np.genfromtxt(data_file, delimiter=',', skip_header=0)
    
    # Filter out rows with indices in indices_to_remove
    filtered_data = np.array([row for i, row in enumerate(data) if i not in indices_to_remove])

    # Generate the output file path
    os.makedirs(output_dir, exist_ok=True)
    if "LCA" in ranking_file:
        suffix = "_LCA"
    else:
        suffix = "_BoostIn"
    column_index = ranking_file.split('_column_')[1].split('_')[0]
    output_file = os.path.join(output_dir, os.path.basename(data_file).replace('.csv', f'_filtered_{percentage}_column{column_index}{suffix}.csv'))

    # Save the filtered data to the output directory
    np.savetxt(output_file, filtered_data, delimiter=',', fmt='%s')
    print(f"Filtered data saved to: {output_file}")

# Main function to handle argument parsing and execution
def main():
    """
    Main function to parse arguments and execute the filtering process.
    """
    parser = argparse.ArgumentParser(description="Filter top-ranked samples from a dataset.")
    parser.add_argument("ranking_file", type=str, help="Path to the ranking output file.")
    parser.add_argument("data_file", type=str, help="Path to the dataset CSV file (in ../data/ directory).")
    parser.add_argument("percentage", type=str, help="Percentage of top-ranked samples to remove (e.g., '1%').")
    parser.add_argument("output_dirID", type=str, help="Unique name of output dir")
    
    args = parser.parse_args()

    # Call the filter_data function with parsed arguments
    rank_file_path = f"../influence_scores/{args.ranking_file}"
    data_file_path = f"../../data/{args.data_file}"
    output_path = f"../custom_data/{args.output_dirID}/"
    filter_data(rank_file_path, data_file_path, args.percentage, output_path)

if __name__ == "__main__":
    main()
