import numpy as np

def sum_second_column(file_path):
    try:
        # Load the file into a NumPy array
        data = np.loadtxt(file_path)
        
        # Ensure the file has at least two columns
        if data.shape[1] < 2:
            raise ValueError("The file must contain at least two columns.")
        
        # Sum the values in the second column (column index 1)
        total = np.sum(data[:, 1])
        return total
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sum_second_column_numpy.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    result = sum_second_column(file_path)
    if result is not None:
        print(f"The sum of the values in the second column is: {result}")
