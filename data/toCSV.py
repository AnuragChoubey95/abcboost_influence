import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define file paths
input_file = "navalplantmaintenance.csv"
intermediate_file = "navalplantmaintenance_comma_separated.csv"
processed_file = "navalplantmaintenance_processed.csv"

# Step 1: Convert Space-Separated Values into Proper CSV Format with Commas
with open(input_file, "r") as infile, open(intermediate_file, "w") as outfile:
    for line in infile:
        # Replace multiple spaces/tabs with a single comma
        formatted_line = ",".join(line.split())
        outfile.write(formatted_line + "\n")

print(f"Converted space-separated file to comma-separated format: {intermediate_file}")

# Step 2: Load the Newly Formatted CSV
df = pd.read_csv(intermediate_file, header=None)

# Step 3: Remove the Last Row
df = df.iloc[:-1, :]

# Step 4: Remove the Last Column Completely
df = df.iloc[:, :-1]  # Drop the last column

# Step 5: Move the Modified Last Column to the First Position
last_column = df.columns[-1]  # Get the current last column index
df = df[[last_column] + [col for col in df.columns if col != last_column]]

# Save the processed dataset
df.to_csv(processed_file, index=False)
print(f"Processed dataset saved as: {processed_file}")

# Step 6: Train-Test Split (50/50 with Seed 42)
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Save Train & Test Datasets
train_file = "naval.train.csv"
test_file = "naval.test.csv"
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Train dataset saved to: {train_file}")
print(f"Test dataset saved to: {test_file}")
print(f"\nTrain-Test Split Summary:")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
