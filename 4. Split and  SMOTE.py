import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- 1. LOAD DATA ---
# Using raw string r'' for path to avoid errors
csv_path = r'C:\College code\Major Project\DataSets\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv'
df = pd.read_csv(csv_path)

# Clean column names (remove hidden spaces)
df.columns = df.columns.str.strip()

# --- 2. CRITICAL FIX: DROP HIGH-CARDINALITY COLUMNS ---
# You MUST drop these before get_dummies, or your RAM will crash.
cols_to_drop = [
    'Flow ID', 
    'Source IP', 
    'Destination IP', 
    'Timestamp', 
    'SimillarHTTP'
]

# Only drop columns that actually exist in the dataframe
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Dropped columns to save memory: {existing_cols_to_drop}")

# --- 3. CLEANING ---
# Handle Infinity and NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define Target (y) and Features (X)
# Map 'BENIGN' to 0, everything else to 1
y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
X = df.drop(columns=['Label'])

# --- 4. ENCODING ---
# Now get_dummies is safe because we removed the IPs and IDs
print("Encoding categorical features...")
X = pd.get_dummies(X, drop_first=True)

print(f"New Data Shape: {X.shape}") 
# Shape should now be manageable (e.g., ~80 columns, not 200,000)

# --- 5. SCALING ---
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. SPLIT & SMOTE ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("Applying SMOTE... (this might take a moment)")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nOriginal Training Set size: {X_train.shape[0]}")
print(f"SMOTE Training Set size: {X_train_smote.shape[0]}")
print(f"SMOTE Training Attack count (1): {np.sum(y_train_smote)}")