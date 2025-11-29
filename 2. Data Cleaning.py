import pandas as pd
import numpy as np

## Load the dataset
# Replace 'your_dataset.csv' with the actual path to your data file
df = pd.read_csv('C:\\College code\\Major Project\\DataSets\\TrafficLabelling\\Wednesday-workingHours.pcap_ISCX.csv')

## 2.1 Clean Column Names and Drop Unnecessary Columns
# Clean up columns by stripping whitespace and replacing special chars if needed
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

# Drop columns that are irrelevant (e.g., flow IDs, timestamps, or constant/near-constant features)
# This step is often based on in-depth EDA; here are common drops for CIC-IDS2017
cols_to_drop = ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'SimillarHTTP'] # Add more based on your dataset
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

## 2.2 Handle Infinity and NaN Values
# Replace infinite values with NaN, then drop rows with any NaN.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

## 2.3 Remove Duplicates (Optional but recommended)
df.drop_duplicates(inplace=True)

print(f"Dataset shape after cleaning: {df.shape}")