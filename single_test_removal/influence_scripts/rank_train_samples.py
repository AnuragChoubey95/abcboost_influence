import numpy as np
import argparse
import os

def process_infl_scores(input_csv, column_index, output_file):
    """
    Processes the influence scores from a CSV file and ranks rows based on a specific column.
    
    Args:
        input_csv (str): Path to the input CSV file.
        column_index (int): Index of the column to rank the rows by.
        output_file (str): Path to the output file for ranked indices and scores.
    """
    file_path = os.path.join("../influence_scores/", input_csv)

    try:
        # Load CSV data and handle invalid/missing entries
        data = np.genfromtxt(file_path, delimiter=',', skip_header=0, dtype=float)
    
        # Print the filename and dimensions
        print(f"Processing file: {file_path}")
        print(f"Data dimensions: {data.shape}")
    
        # Print last column
        print("Last column:")
        print(data[:, -1])

    except ValueError as e:
        print(f"\nError reading file: {file_path}")
        print(f"{str(e)}")
        print("Possible causes: missing/extra columns, incorrect delimiter, or corrupted file.")
        return  # Exit function to avoid further errors

    # Check for invalid or missing data
    if np.isnan(data).any():
        print(f"Warning: Missing or invalid data found in {file_path}. Replacing NaNs with 0.")
        data = np.nan_to_num(data, nan=0.0)

    # Ensure the column index is within range
    if column_index < 0 or column_index >= data.shape[1]:
        raise ValueError(f"Column index {column_index} is out of range. Valid range is 0 to {data.shape[1] - 1}.")

    # Extract the specified column
    column_values = data[:, column_index]

    # Rank rows based on most positive to most negative
    ranked_indices = np.argsort(-column_values)

    # Save the ranked indices and their scores to the output file
    output_path = os.path.join("../influence_scores/", output_file)
    with open(output_path, 'w') as f:
        f.write("RowIndex,ColumnValue\n")
        for index in ranked_indices:
            f.write(f"{index},{column_values[index]:.6f}\n")

    print(f"Ranked rows by column {column_index} have been saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Processinfluence scores and rank rows by a specific column.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing influence scores.")
    parser.add_argument("output_prefix", type=str, help="Prefix for the output file. The column index will be appended.")
    parser.add_argument("column_index", type=int, help="Column index to rank the rows by (0-based index).")

    args = parser.parse_args()

    # Generate the output filename based on the column index and input CSV content
    if "LCA" in args.input_csv:
        suffix = "_LCA"
    else:
        suffix = "_BoostIn"
    output_file = f"{args.output_prefix}_column_{args.column_index}{suffix}.csv"

    # Process the BoostIn scores
    process_infl_scores(args.input_csv, args.column_index, output_file)
    print(f"Ranked rows by column {args.column_index} have been saved to {output_file}")

if __name__ == "__main__":
    main()
