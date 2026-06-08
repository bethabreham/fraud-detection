import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the processed fraud data
df = pd.read_csv('data/processed/fraud_data_processed.csv')
print(f"Loaded data shape: {df.shape}")

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nBefore SMOTE:")
print(f"  Train shape: {X_train.shape}")
print(f"  Train fraud count: {y_train.sum()}")
print(f"  Train fraud percentage: {y_train.mean()*100:.4f}%")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Train shape: {X_train_resampled.shape}")
print(f"  Train fraud count: {y_train_resampled.sum()}")
print(f"  Train fraud percentage: {y_train_resampled.mean()*100:.4f}%")

# Save resampled data
resampled_df = pd.DataFrame(X_train_resampled, columns=X.columns)
resampled_df['class'] = y_train_resampled
resampled_df.to_csv('data/processed/fraud_data_resampled.csv', index=False)
print("\nSaved resampled data to data/processed/fraud_data_resampled.csv")

# Save test data
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['class'] = y_test
test_df.to_csv('data/processed/fraud_data_test.csv', index=False)
print("Saved test data to data/processed/fraud_data_test.csv")