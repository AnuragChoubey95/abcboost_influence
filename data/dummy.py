import pandas as pd

# Modify and overwrite german.train
train_df = pd.read_csv("german.train.csv", header=None)
train_df[0] = train_df[0].replace({2: 1, 1: 0})
train_df.to_csv("german.train.csv", index=False, header=False)

# Modify and overwrite german.test
test_df = pd.read_csv("german.test.csv", header=None)
test_df[0] = test_df[0].replace({2: 1, 1: 0})
test_df.to_csv("german.test.csv", index=False, header=False)
