import pandas as pd
from sklearn.preprocessing import StandardScaler

# If df is not already defined in the environment, load it from a CSV file.
if 'df' not in globals():
    # Note: Use raw string (r'...') for paths to avoid issues with backslashes
    df = pd.read_csv(r'C:\College code\Major Project\DataSets\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv')

# --- FIX START: Clean Column Names ---
# This removes spaces from the start/end of column names (e.g., ' Label ' becomes 'Label')
df.columns = df.columns.str.strip()
# --- FIX END ---

# Optional: Print columns to verify the name is now correct
print("Columns found:", df.columns.tolist())

# 3.1 Binary Target Encoding
label_col = 'Label' 

# Create the binary target variable (y)
y = df[label_col].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)

# Features (X) - Drop the original label column
X = df.drop(columns=[label_col])

# --- Handle remaining Categorical Features ---
X = pd.get_dummies(X, drop_first=True)

# 3.2 Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled successfully.")