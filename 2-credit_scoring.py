# credit_scoring.py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd

# The URL for the raw German Credit Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# These are the actual column names from the dataset documentation
columns = [
    'status', 'duration', 'credit_history', 'purpose', 'amount', 
    'savings', 'employment_duration', 'installment_rate', 'personal_status_sex', 
    'other_debtors', 'residence_since', 'property', 'age', 
    'other_installment_plans', 'housing', 'number_credits', 
    'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk'
]

# Load the data (it's space-separated)
df = pd.read_csv(url, sep=' ', names=columns)

# Map the credit_risk column: 1 is 'Good', 2 is 'Bad'
# (Crucial for later: Machine Learning likes 0 and 1)
df['credit_risk'] = df['credit_risk'].replace({1: 1, 2: 0})

print("Data successfully loaded directly from UCI!")
# print(df.head(10))

import seaborn as sns
import matplotlib.pyplot as plt

# Check how many Good (1) vs Bad (0) cases we have
print(df['credit_risk'].value_counts())

# sns.countplot(x='credit_risk', data=df, palette='viridis')
# plt.title('Distribution of Credit Risk (1=Good, 0=Bad)')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='credit_risk', y='age', data=df)
# plt.title('Age vs Credit Risk')
# plt.xticks([0, 1], ['Bad (0)', 'Good (1)'])
# plt.show()


# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='duration', y='amount', hue='credit_risk', data=df, alpha=0.6)
# plt.title('Loan Duration vs Amount (Colored by Risk)')
# plt.show()


from sklearn.preprocessing import LabelEncoder

# Initialize the encoder
le = LabelEncoder()

# Identify columns that are 'object' type (text/strings)
categorical_cols = df.select_dtypes(include=['object']).columns

# Loop through and transform each one
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Data Preprocessing Complete!")
# print(df.head()) # Now you will see numbers like 0, 1, 2 instead of A11, A43...


from sklearn.model_selection import train_test_split

# Define Features (X) and Target (y)
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model
model.fit(X_train, y_train)

# 3. Make predictions on the "Exam" (Test set)
y_pred = model.predict(X_test)

# 4. Check results
print("="*30)
print("Result:")
print("="*30)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


import numpy as np

# Get importance levels
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the top 5
print("Top 5 Risk Factors:")
for i in range(5):
    print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")