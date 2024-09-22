import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
# from xgboost import XGBClassifier


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        metadata = file.readline().strip()  # Read the first line (metadata)
        file.readline()  # Skip the empty second line
        df = pd.read_csv(file, sep='\t')  # Read the rest of the file
    return df, metadata


# method to preprocess data
def preprocess_dataframe(df):
    # Convert columns to appropriate data types
    df['PPG'] = pd.to_numeric(df['PPG'], errors='coerce')
    df['Derivative'] = pd.to_numeric(df['Derivative'], errors='coerce')

    # Handle 'Time' column
    df['Time'] = df['Time'].apply(lambda x: pd.to_numeric(
        x.replace('"', '') if isinstance(x, str) else x, errors='coerce'))

    # Handle 'Laterality' column
    df['Laterality'] = df['Laterality'].apply(
        lambda x: x.replace('"', '') if isinstance(x, str) else x)

    # Remove rows where PPG is 0.0
    df = df[df['PPG'] != 0.0].copy()

    # # Reset the index after removing rows
    # df.reset_index(drop=True, inplace=True)

    df = df.dropna()  # Remove any rows with NaN values

    return df


def combine_data_min_count(df_L, df_R):
    # Filter left arm data
    df_L_filtered = df_L[df_L['Laterality'] == 'LEFT_ARM']

    # Filter right arm data
    df_R_filtered = df_R[df_R['Laterality'] == 'RIGHT_ARM']

    # Get the minimum count
    min_count = min(len(df_L_filtered), len(df_R_filtered))

    # Truncate both dataframes to the minimum count
    df_L_truncated = df_L_filtered.head(min_count)
    df_R_truncated = df_R_filtered.head(min_count)

    # Combine the truncated datasets
    df_combined = pd.concat(
        [df_L_truncated, df_R_truncated], ignore_index=True)

    return df_combined


# Read the CSV files
df_L, metadata_left = read_csv_file('train/leftT.csv')
df_R, metadata_right = read_csv_file('train/rightT.csv')
df, metadata_none = read_csv_file('train/noT.csv')

# Preprocess the dataframes
df_L = preprocess_dataframe(df_L)
df_R = preprocess_dataframe(df_R)
df = preprocess_dataframe(df)

# Combine data using minimum count method
df_combined_min = combine_data_min_count(df_L, df_R)

# Add a column to indicate this is tourniquet data
df_combined_min['is_tourniquet'] = 1
# Add a column to indicate this is not tourniquet data
df['is_tourniquet'] = 0

# Convert 'Laterality' to numeric (0 for RIGHT_ARM, 1 for LEFT_ARM) -- changes feature to be numeric(best practice)
df['Laterality'] = (df['Laterality'] == 'LEFT_ARM').astype(int)
df_combined_min['Laterality'] = (
    df_combined_min['Laterality'] == 'LEFT_ARM').astype(int)

# Combine tourniquet and non-tourniquet data
df_all = pd.concat([df_combined_min, df], ignore_index=True)

# Shuffle the combined dataframe
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)


# Prepare features and target before feature engineering
X = df_all[['Time', 'PPG', 'Derivative', 'Laterality']]
y = df_all['is_tourniquet']

# Split the data, stratifying by the target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define a function for feature engineering that doesn't use future information


def feature_engineer_no_leakage(df):
    # Sort by Time within each Laterality group
    df = df.sort_values(['Laterality', 'Time'])

    # Use only past information for rolling statistics
    df['PPG_rolling_mean'] = df.groupby('Laterality')['PPG'].rolling(
        window=10, min_periods=1).mean().reset_index(0, drop=True)
    df['Derivative_rolling_mean'] = df.groupby('Laterality')['Derivative'].rolling(
        window=10, min_periods=1).mean().reset_index(0, drop=True)

    # Use shift for rate of change to avoid using future information
    df['PPG_rate_of_change'] = df.groupby(
        'Laterality')['PPG'].diff() / df.groupby('Laterality')['Time'].diff()
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    return df  # return changes


# Apply feature engineering separately to train and test
X_train = feature_engineer_no_leakage(X_train)
X_test = feature_engineer_no_leakage(X_test)

# Now select features for modeling
features = ['PPG', 'Derivative', 'PPG_rolling_mean',
            'Derivative_rolling_mean', 'PPG_rate_of_change', 'Laterality']
X_train = X_train[features]
X_test = X_test[features]


# print("Combined data (minimum count method):")
print(X_train)

print(X_test)

# scalar
# test model

# Train and evaluate XGBoost model
# xgb_model = XGBClassifier(enable_categorical=True, random_state=42)
# xgb_model.fit(X_train, y_train)

# # Make predictions
# y_pred_xgb = xgb_model.predict(X_test)

# # Evaluate XGBoost model
# print("XGBoost Results:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
# print(classification_report(y_test, y_pred_xgb))
