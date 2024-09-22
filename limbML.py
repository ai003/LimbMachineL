import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time


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

print("Combined data (minimum count method):")
print(df_combined_min)

print(df)
