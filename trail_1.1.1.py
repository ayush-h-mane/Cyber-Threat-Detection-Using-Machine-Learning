import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE

# --- 1. CONFIGURATION ---
# This is the path you provided. Make sure the file exists here.
csv_path = r'C:\College code\Major Project\DataSets\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv'

print("Loading dataset...")
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    print("Please check the path and try again.")
    exit()

# --- 2. DATA CLEANING ---
print("Cleaning data...")
# Clean column names (remove hidden spaces)
df.columns = df.columns.str.strip()

# DROP HIGH-CARDINALITY COLUMNS (Crucial to prevent Memory Crash)
# We drop IPs, IDs, and Timestamps because they confuse the model and take up too much RAM
cols_to_drop = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 'SimillarHTTP']
# Only drop columns that actually exist in the dataframe
existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=existing_drop_cols)

# Handle NaNs and Infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define X and y
print("Encoding target labels...")
# Map 'BENIGN' to 0, everything else to 1
y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
X = df.drop(columns=['Label'])

# Encode Categorical Features
print("One-Hot Encoding features...")
X = pd.get_dummies(X, drop_first=True)

# SAVE COLUMN NAMES
# The web app needs this list to ensure user input matches the model's expected shape
model_columns = list(X.columns)
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)
print(f"Saved 'model_columns.pkl' ({len(model_columns)} features)")

# Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SAVE SCALER
# The web app needs this to scale user input exactly like the training data
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved 'scaler.pkl'")

# --- 3. DATA SPLITTING & BALANCING ---
print("Splitting data and applying SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle Imbalance (Make the model learn attacks better)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- 4. TRAIN & SAVE RANDOM FOREST ---
print("Training Random Forest (Supervised)...")
# We use n_estimators=50 for speed; you can increase to 100 for better accuracy
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_smote, y_train_smote)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
print("Saved 'rf_model.pkl'")

# --- 5. TRAIN & SAVE ISOLATION FOREST ---
print("Training Isolation Forest (Unsupervised)...")
# Filter only normal traffic for training the anomaly detector
normal_indices = np.where(y_train == 0)[0]
X_train_normal = X_train[normal_indices]

iso_forest = IsolationForest(n_estimators=50, contamination='auto', random_state=42, n_jobs=-1)
iso_forest.fit(X_train_normal)

with open('iso_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
print("Saved 'iso_forest.pkl'")

print("\nSUCCESS: All 4 model files have been saved!")
print("You can now refresh your website.")