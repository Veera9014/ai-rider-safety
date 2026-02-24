import pandas as pd

# Load dataset
data = pd.read_csv("rider_safety_data.csv")

# Show first rows
print("First 5 rows:")
print(data.head())

# Dataset info
print("\nDataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)
print("\nDataSet Summary:")
print(data.describe())