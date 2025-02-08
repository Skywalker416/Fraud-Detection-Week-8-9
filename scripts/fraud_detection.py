import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define data directory
data_dir = "./data/"  # Update this path if needed

# Load datasets
fraud_data = pd.read_csv(os.path.join(data_dir, "Fraud_Data.csv"))
ip_data = pd.read_csv(os.path.join(data_dir, "IpAddress_to_Country.csv"))
credit_data = pd.read_csv(os.path.join(data_dir, "creditcard.csv"))

# Task 1.1 - Handle Missing Values
fraud_data.dropna(inplace=True)
ip_data.dropna(inplace=True)
credit_data.dropna(inplace=True)

# Task 1.2 - Data Cleaning (Remove Duplicates, Correct Data Types)
fraud_data.drop_duplicates(inplace=True)
ip_data.drop_duplicates(inplace=True)
credit_data.drop_duplicates(inplace=True)

# Convert timestamps to datetime format
fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

# Task 1.3 - Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
sns.countplot(x='class', data=fraud_data)
plt.title("Fraudulent vs Non-Fraudulent Transactions")
plt.show()

# Check distribution of purchase values
plt.figure(figsize=(10, 5))
sns.histplot(fraud_data['purchase_value'], bins=50, kde=True)
plt.title("Distribution of Purchase Values")
plt.show()

# Task 1.4 - Merge IP Address Data
fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)

fraud_data = fraud_data.merge(ip_data, 
                              left_on='ip_address', 
                              right_on='lower_bound_ip_address', 
                              how='left')
fraud_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)

# Task 1.5 - Feature Engineering
fraud_data['transaction_duration'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()
fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek

# Transaction Frequency
fraud_data['user_transaction_count'] = fraud_data.groupby('user_id')['user_id'].transform('count')

# Normalize purchase value
fraud_data['purchase_value_scaled'] = (fraud_data['purchase_value'] - fraud_data['purchase_value'].mean()) / fraud_data['purchase_value'].std()

# Encode categorical features
fraud_data = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)

# Display dataset info
print(fraud_data.info())
print(fraud_data.head())
