import numpy as np
import argparse

def process_boostin_scores(input_csv, column_index, output_file):
    """
    Processes the BoostIn scores from a CSV file and ranks rows based on a specific column.
    
    Args:
        input_csv (str): Path to the input CSV file.
        column_index (int): Index of the column to rank the rows by.
        output_file (str): Path to the output file for ranked indices and scores.
    """
    # Load CSV data and handle invalid/missing entries
    data = np.genfromtxt(input_csv, delimiter=',', skip_header=0)

    # # Print the dimensions of the data
    # print(f"Data dimensions: {data.shape}")
    # print("Last column:")
    # print(data[:, -1])

       # Check for invalid or missing data
    if np.isnan(data).any():
        print("Warning: Missing or invalid data found at the following locations (row, column):")
        nan_locations = np.argwhere(np.isnan(data))
        for loc in nan_locations:
            print(f"Row: {loc[0]}, Column: {loc[1]}")
        print("Replacing NaNs with 0.")
        data = np.nan_to_num(data, nan=0.0)


    # Ensure the column index is within range
    if column_index < 0 or column_index >= data.shape[1]:
        raise ValueError(f"Column index {column_index} is out of range. Valid range is 0 to {data.shape[1] - 1}.")

    # Extract the specified column
    column_values = data[:, column_index]

    # Rank rows based on the specified column in descending order
    ranked_indices = np.argsort(-column_values)

    # Save the ranked indices and their scores to the output file
    with open("../ranked/" + output_file, 'w') as f:
        f.write("RowIndex,ColumnValue\n")
        for index in ranked_indices:
            f.write(f"{index},{column_values[index]:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="Process BoostIn scores and rank rows by a specific column.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing BoostIn scores.")
    parser.add_argument("output_prefix", type=str, help="Prefix for the output file. The column index will be appended.")
    parser.add_argument("column_index", type=int, help="Column index to rank the rows by (0-based index).")

    args = parser.parse_args()

    if "LCA" in args.input_csv:
        suffix = "_LCA"
    else:
        suffix = "_BoostIn"
    output_file = f"{args.output_prefix}_column_{args.column_index}{suffix}.csv"

    # Process the BoostIn scores
    process_boostin_scores(args.input_csv, args.column_index, output_file)
    print(f"Ranked rows by column {args.column_index} have been saved to {output_file}")

if __name__ == "__main__":
    main()
