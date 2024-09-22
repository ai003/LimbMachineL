import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time


# steps read the data from each file
# then separate the data from each dataset then combine
# get features and add correct labels
# then use for training
# then test on new data
# then get results
# output graphs for visuals

# Load your CSV file
dataLeft = pd.read_csv('train/leftT.csv')

# Load your CSV file
dataRight = pd.read_csv('train/rightT.csv')

print(dataLeft.info())
print(dataRight.info())

# Check for missing values
print(dataLeft.isnull().sum())

# Check for missing values
print(dataRight.isnull().sum())


# Read the file
with open('armdata.csv', 'r') as file:
    metadata = file.readline().strip()  # Read the first line (metadata)
    file.readline()  # Skip the empty second line
    df = pd.read_csv(file, sep='\t')  # Read the rest of the file

# Print metadata
print("Metadata:", metadata)

# Convert columns to appropriate data types
df['PPG'] = pd.to_numeric(df['PPG'], errors='coerce')
df['Derivative'] = pd.to_numeric(df['Derivative'], errors='coerce')

# Handle 'Time' column
df['Time'] = df['Time'].apply(lambda x: pd.to_numeric(
    x.replace('"', '') if isinstance(x, str) else x, errors='coerce'))

# Handle 'Laterality' column
df['Laterality'] = df['Laterality'].apply(
    lambda x: x.replace('"', '') if isinstance(x, str) else x)

# Add a column to indicate this is tourniquet data
df['is_tourniquet'] = 1

# # Separate data by arm
# df_left = df[df['Laterality'] == 'LEFT_ARM'].copy()
# df_right = df[df['Laterality'] == 'RIGHT_ARM'].copy()

# # Create relative time features and set as index
# df_left['Relative_Time'] = df_left['Time'] - df_left['Time'].min()
# df_right['Relative_Time'] = df_right['Time'] - df_right['Time'].min()
# df_left.set_index('Relative_Time', inplace=True)
# df_right.set_index('Relative_Time', inplace=True)


# # Function to calculate features
# def calculate_features(df):
#     df['PPG_rolling_mean'] = df['PPG'].rolling(window=10).mean()
#     df['Derivative_rolling_mean'] = df['Derivative'].rolling(window=10).mean()
#     df['PPG_rate_of_change'] = df['PPG'].diff() / df.index.to_series().diff()
#     return df


# # Calculate features for each arm
# df_left = calculate_features(df_left)
# df_right = calculate_features(df_right)

# # Function for model training and evaluation


# def train_and_evaluate(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     model = LogisticRegression(random_state=42)

#     start_time = time.time()
#     model.fit(X_train_scaled, y_train)
#     training_time = time.time() - start_time

#     start_time = time.time()
#     y_pred = model.predict(X_test_scaled)
#     prediction_time = time.time() - start_time

#     accuracy = accuracy_score(y_test, y_pred)

#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Training Time: {training_time:.4f} seconds")
#     print(f"Prediction Time: {prediction_time:.4f} seconds")
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

#     # Feature importance (coefficients for logistic regression)
#     feature_importance = pd.DataFrame({
#         'feature': X.columns,
#         'importance': abs(model.coef_[0])
#     }).sort_values('importance', ascending=False)

#     print("\nFeature Importance:")
#     print(feature_importance)

#     return model, feature_importance


# # Analyze each arm separately
# for df, arm in zip([df_left, df_right], ['Left', 'Right']):
#     print(f"\nAnalysis for {arm} Arm:")
#     X = df[['PPG', 'Derivative', 'PPG_rolling_mean',
#             'Derivative_rolling_mean', 'PPG_rate_of_change']]
#     y = df['is_tourniquet']
#     model, feature_importance = train_and_evaluate(X, y)

#     # Visualize feature importance
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=feature_importance)
#     plt.title(f'Feature Importance for {arm} Arm')
#     plt.tight_layout()
#     plt.show()

# Visualize feature importance for combined analysis
# plt.figure(figsize=(10, 6))
# sns.barplot(x='importance', y='feature', data=feature_importance)
# plt.title('Feature Importance for Combined Analysis')
# plt.tight_layout()
# plt.show()

df.set_index('Time', inplace=True)
# plt.figure(figsize=(12, 6))
# plt.plot(df.index, df['PPG'])
# plt.title('PPG Over Time')
# plt.xlabel('Time')
# plt.ylabel('PPG Value')
# plt.show()
# df.reset_index(inplace=True)

# # Display info about the DataFrame
# print(df.info())

# mean over sliding window of 10
df['PPG_rolling_mean'] = df.groupby('Laterality')['PPG'].rolling(
    window=10).mean().reset_index(0, drop=True)

# deriv over same
df['Derivative_rolling_mean'] = df.groupby('Laterality')['Derivative'].rolling(
    window=10).mean().reset_index(0, drop=True)


# Calculate rate of change
df['PPG_rate_of_change'] = df.groupby(
    'Laterality')['PPG'].diff() / df.index.to_series().diff()

# see the difference
df['PPG_derivative'] = df['PPG'].diff()
# df['PPG_rolling_mean'] = df['PPG'].rolling(window=10).mean()

# Create a figure with two subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 8))

# Loop through the unique values in 'Laterality'
for i, laterality in enumerate(df['Laterality'].unique()):
    # Filter the DataFrame for the current laterality
    group = df[df['Laterality'] == laterality]

    # Plot Rolling Mean and Rolling Derivative for the current group
    axs[i].plot(group['PPG_rolling_mean'],
                label='Rolling Mean (Window=10)', color='blue')
    axs[i].plot(group['Derivative_rolling_mean'],
                label='Rolling Derivative', color='orange')

    # Customize the plot for the current laterality
    axs[i].set_title(f'PPG Rolling Mean and Rolling Derivative - {laterality}')
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('Values')
    axs[i].legend()

# Subplot for Direct Derivative
# Use the last subplot for the direct derivative
axs[2].plot(df['PPG_derivative'], label='Direct Derivative', color='green')
axs[2].plot(group['PPG_rolling_mean'],
            label='Rolling Mean (Window=10)', color='blue')
axs[2].set_title('Direct Derivative of PPG')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Values')
axs[2].legend()

# plt.tight_layout()
# plt.show()

X = df[['PPG', 'Derivative', 'PPG_rolling_mean',
        'Derivative_rolling_mean', 'PPG_rate_of_change']]
y = df['is_tourniquet']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Check for missing values
print(X_train.isnull().sum())

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
