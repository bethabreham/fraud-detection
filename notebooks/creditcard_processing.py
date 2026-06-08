import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load credit card data
df_credit = pd.read_csv('data/raw/creditcard.csv')
print(f"Credit card data shape: {df_credit.shape}")
print(df_credit.head())

# Check missing values
print("\nMissing values:")
print(df_credit.isnull().sum())

# Check class imbalance
print("\nClass distribution:")
print(df_credit['Class'].value_counts())
print(f"Fraud percentage: {df_credit['Class'].mean()*100:.4f}%")

# Scale Amount feature (V1-V28 are already scaled)
scaler = StandardScaler()
df_credit['Amount_scaled'] = scaler.fit_transform(df_credit[['Amount']])

# Drop original Amount column
df_credit = df_credit.drop('Amount', axis=1)

print(f"\nFinal credit card shape: {df_credit.shape}")
print(f"Columns: {df_credit.columns.tolist()}")

# Save processed data
df_credit.to_csv('data/processed/creditcard_processed.csv', index=False)
print("\nSaved to ../data/processed/creditcard_processed.csv")