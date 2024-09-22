import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Read the tourniquet data
with open('armdata.csv', 'r') as file:
    metadata = file.readline().strip()  # Read the first line (metadata)
    file.readline()  # Skip the empty second line
    df_tourniquet = pd.read_csv(file, sep='\t')  # Read the rest of the file

print("Metadata:", metadata)

# Preprocess tourniquet data
df_tourniquet['PPG'] = pd.to_numeric(df_tourniquet['PPG'], errors='coerce')
df_tourniquet['Derivative'] = pd.to_numeric(
    df_tourniquet['Derivative'], errors='coerce')
df_tourniquet['Time'] = pd.to_numeric(df_tourniquet['Time'].apply(
    lambda x: x.replace('"', '') if isinstance(x, str) else x), errors='coerce')
df_tourniquet['Laterality'] = df_tourniquet['Laterality'].apply(
    lambda x: x.replace('"', '') if isinstance(x, str) else x)
df_tourniquet['is_tourniquet'] = 1  # Label tourniquet data as 1

# Generate synthetic non-tourniquet data (you should replace this with real non-tourniquet data if available)
df_non_tourniquet = pd.DataFrame({
    'PPG': np.random.normal(df_tourniquet['PPG'].mean(), df_tourniquet['PPG'].std(), len(df_tourniquet)),
    'Derivative': np.random.normal(df_tourniquet['Derivative'].mean(), df_tourniquet['Derivative'].std(), len(df_tourniquet)),
    'Time': np.random.uniform(df_tourniquet['Time'].min(), df_tourniquet['Time'].max(), len(df_tourniquet)),
    'Laterality': np.random.choice(['LEFT_ARM', 'RIGHT_ARM'], len(df_tourniquet)),
    'is_tourniquet': 0  # Label non-tourniquet data as 0
})

# Combine tourniquet and non-tourniquet data
df = pd.concat([df_tourniquet, df_non_tourniquet], ignore_index=True)

# Shuffle the combined dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create features


def calculate_features(df):
    df['PPG_rolling_mean'] = df.groupby('Laterality')['PPG'].rolling(
        window=10).mean().reset_index(0, drop=True)
    df['Derivative_rolling_mean'] = df.groupby('Laterality')['Derivative'].rolling(
        window=10).mean().reset_index(0, drop=True)
    df['PPG_rate_of_change'] = df.groupby(
        'Laterality')['PPG'].diff() / df.groupby('Laterality')['Time'].diff()
    return df


df = calculate_features(df)

# Prepare data for modeling
X = df[['PPG', 'Derivative', 'PPG_rolling_mean',
        'Derivative_rolling_mean', 'PPG_rate_of_change']]
y = df['is_tourniquet']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(pipeline.named_steps['classifier'].coef_[0])
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Tourniquet Classification')
plt.tight_layout()
plt.show()

# Print sample rows
print("\nSample rows from the dataset:")
print(df.head())

# Print class distribution
print("\nClass Distribution:")
print(df['is_tourniquet'].value_counts(normalize=True))
