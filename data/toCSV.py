import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from pprint import pprint

# Fetch dataset
combined_cycle_power_plant = fetch_ucirepo(id=294) 

# Data (as pandas DataFrames)
X = combined_cycle_power_plant.data.features 
y = combined_cycle_power_plant.data.targets

# Pretty-print metadata and variables
print("=== METADATA ===")
pprint(combined_cycle_power_plant.metadata)

print("\n=== VARIABLES ===")
pprint(combined_cycle_power_plant.variables)

# Create a 50/50 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.5, 
    random_state=42
)

# Combine features and targets for each split
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)

# Save each split to CSV
train_df.to_csv("combined_cycle_power_plant.train.csv", index=False)
test_df.to_csv("combined_cycle_power_plant.test.csv", index=False)

print("\nTrain (combined_cycle_power_plant.train.csv) and test (combined_cycle_power_plant.test.csv) files have been saved.")
