import pandas as pd

# Function to move the last column to the first position
def move_last_column_first(file_path):
    # Read CSV
    df = pd.read_csv(file_path)

    # Move the last column to the first
    last_col = df.columns[-1]  # Get last column name
    cols = [last_col] + [col for col in df.columns if col != last_col]  # Reorder
    df = df[cols]

    # Save the modified file
    df.to_csv(file_path, index=False)
    print(f"✅ Processed: {file_path}")

# File paths
train_file = "surgical.train.csv"
test_file = "surgical.test.csv"

# Apply transformation
move_last_column_first(train_file)
move_last_column_first(test_file)

print("✅ Reordering completed for both train and test files!")
