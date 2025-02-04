from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import logging

# Configure logging
log_file = "bank_marketing_processing.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Starting Bank Marketing dataset processing...")

# Step 1: Fetch Dataset
bank_marketing = fetch_ucirepo(id=222)
logging.info("Dataset fetched successfully.")

# Step 2: Convert Data to Pandas DataFrame
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Log dataset shape
logging.info(f"Features shape: {X.shape}")
logging.info(f"Target shape: {y.shape}")

# Step 3: Combine Features and Target into a Single DataFrame
df = pd.concat([y, X], axis=1)  # Move target column to the first position
logging.info("Moved target column to the first position.")

# Step 4: Identify Categorical Features
categorical_columns = df.select_dtypes(include=['object']).columns
logging.info(f"Identified categorical columns: {list(categorical_columns)}")

# Step 5: Convert Categorical Features into Numerical Using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    logging.info(f"Encoded column: {col}")

logging.info("Converted all categorical features into numerical.")

# Step 6: Handle Missing Data with KNN Imputation
logging.info("Handling missing data using KNN Imputer.")
imputer = KNNImputer(n_neighbors=5)
df[:] = imputer.fit_transform(df)
logging.info("Missing values imputed successfully.")

# Step 7: Train-Test Split (50/50 with Seed 42)
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)
logging.info("Performed 50/50 train-test split.")

# Log dataset shapes
logging.info(f"Train dataset shape: {train_df.shape}")
logging.info(f"Test dataset shape: {test_df.shape}")

# Step 8: Save Train & Test Datasets
train_file = "bank_marketing.train.csv"
test_file = "bank_marketing.test.csv"
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

logging.info(f"Train dataset saved to: {train_file}")
logging.info(f"Test dataset saved to: {test_file}")
logging.info("Bank Marketing dataset processing completed successfully.")

# Print Summary
print(f"Train dataset saved to: {train_file}")
print(f"Test dataset saved to: {test_file}")
print(f"\nTrain-Test Split Summary:")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print(f"\nProcessing log saved to: {log_file}")
