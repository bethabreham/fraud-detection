# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project builds fraud detection models for two transaction streams:
1. E-commerce transactions (with user/device/behavioral data)
2. Bank credit card transactions (anonymized PCA features)

## Dataset Links
- Fraud_Data.csv: https://drive.google.com/file/d/1BCP10YNCMxzJzNjRoUYcxqBk3EHSz7wf/view
- IpAddress_to_Country.csv: https://drive.google.com/file/d/1OagS1rraLIX5wtq8yilXt8gXBcb8fKoC/view
- creditcard.csv: https://drive.google.com/file/d/1Nd6bmQuFq_-RQGVXOgNDQBNiPaZl7KH4/view

## Setup Instructions

1. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

## Project Structure

fraud-detection/
├── .github/workflows/
│   └── unittests.yml
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/
├── tests/
├── models/
├── scripts/
├── .gitignore
├── requirements.txt
└── README.md

## Key Findings (EDA)

### E-commerce Dataset
- Total rows: ~150,000
- Fraud percentage: ~6.7%
- No missing values
- Purchase values are right-skewed with outliers

### Credit Card Dataset
- Total rows: ~284,807
- Fraud percentage: ~0.17% (highly imbalanced)
- No missing values
- V1-V28 are PCA-transformed and already scaled

## Feature Engineering

### IP-to-Country Mapping
- Converted IP addresses to integers
- Used merge_asof for range-based lookup
- Added country column to e-commerce data

### Time-Based Features
- time_since_signup: Hours between signup and purchase
- purchase_hour: Hour of day (0-23)
- purchase_dayofweek: Day of week (0-6)

### Categorical Encoding
- One-hot encoded: source, browser, sex, country

## Class Imbalance Handling

### SMOTE (Synthetic Minority Over-sampling)
- Applied only to training set (never to test set)
- Creates synthetic fraud examples by interpolation
- Balances class distribution to 50/50

### Evaluation Metrics
- Primary: AUC-PR, F1-Score, Recall
- Secondary: Confusion Matrix, Precision

## Model Performance (Task 2)

| Model | F1-Score | AUC-PR | Recall |
|-------|----------|--------|--------|
| Logistic Regression | X.XX | X.XX | X.XX |
| Random Forest | X.XX | X.XX | X.XX |
| XGBoost | X.XX | X.XX | X.XX |

Best Model: [Model Name]

## SHAP Explainability (Task 3)

### Top 5 Fraud Drivers
1. Feature name - Explanation
2. Feature name - Explanation
3. Feature name - Explanation
4. Feature name - Explanation
5. Feature name - Explanation

### Business Recommendations
1. Recommendation based on SHAP insights
2. Recommendation based on SHAP insights
3. Recommendation based on SHAP insights

## CI/CD
GitHub Actions runs on every push to main:
- Python 3.13 setup
- Dependency installation
- Linting with flake8
- Unit tests with pytest

## Dependencies
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm
- shap, imbalanced-learn
- pytest, flake8

## Author
Beth Abreham - 10 Academy KAIM 9 Cohort