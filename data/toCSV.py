from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Fetch dataset
dry_bean = fetch_ucirepo(id=602)

# Step 2: Convert data to pandas DataFrames
X = dry_bean.data.features
y = dry_bean.data.targets

# Step 3: Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)  # Keep the original order (features first, target last)

# Step 4: Print important dataset information
print("🔍 Dataset Information:")
print(f"Shape of the dataset: {df.shape}")
print(f"Column Names: {list(df.columns)}")
print(f"Number of missing values:\n{df.isnull().sum()}")

# Step 5: Convert categorical class labels into numerical labels
label_encoder = LabelEncoder()
target_col = y.columns[0]  # Get the name of the target column from DataFrame
df[target_col] = label_encoder.fit_transform(df[target_col])

# Verify label encoding
print(f"\n🔢 Class Label Encoding:")
for class_label, numeric_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{class_label}: {numeric_label}")

# Step 6: Move the target column (last column) to the first position
columns = [target_col] + [col for col in df.columns if col != target_col]
df = df[columns]

# Step 7: Train-Test Split (50/50 with seed 42)
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Step 8: Print split information
print("\n📊 Train-Test Split Summary:")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")

# Step 9: Save datasets to CSV
train_file = "dry_bean.train.csv"
test_file = "dry_bean.test.csv"
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\n✅ Train dataset saved to: {train_file}")
print(f"✅ Test dataset saved to: {test_file}")
