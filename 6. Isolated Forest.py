import pandas as pd
import numpy as np  # <--- This fixes your current error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# --- STEP 1: LOAD AND PREPROCESS DATA (Required to define X_train, y_train) ---

# Load data
print("Loading dataset...")
df = pd.read_csv(r'C:\College code\Major Project\DataSets\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Drop High-Cardinality Columns (To prevent Memory Error)
drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 'SimillarHTTP']
existing_drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_drop_cols)

# Handle NaNs and Infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Define X and y
label_col = 'Label'
y = df[label_col].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
X = df.drop(columns=[label_col])

# Encode and Scale  
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 2: ISOLATION FOREST (Unsupervised Anomaly Detection) ---

print("Filtering normal traffic for training...")
# Filter the training data to include ONLY normal (BENIGN) traffic
# We convert y_train to a numpy array to ensure indexing works correctly
y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
normal_traffic_indices = np.where(y_train_np == 0)[0]
X_train_normal = X_train[normal_traffic_indices]

print(f"Training Isolation Forest on {len(X_train_normal)} normal samples...")

# Initialize and train the Isolation Forest model
iso_forest = IsolationForest(
    n_estimators=100, 
    contamination='auto', 
    random_state=42, 
    n_jobs=-1
)
iso_forest.fit(X_train_normal)

# Predict anomaly scores for the UNTOUCHED test set (X_test)
print("Predicting on test set...")
iso_predictions_binary = iso_forest.predict(X_test)

# Map Isolation Forest output (-1, 1) to (1, 0) for consistency (Anomaly=1, Normal=0)
# Isolation Forest returns -1 for anomalies, 1 for normal.
# We want 1 for anomalies (to match our Attack label), 0 for normal.
iso_predictions_mapped = np.where(iso_predictions_binary == -1, 1, 0)

# Compare the Isolation Forest Anomaly detections against true labels
print("\n--- Isolation Forest (Unsupervised) Anomaly Detection Summary ---")
print(f"Total Test Samples: {len(y_test)}")
print(f"Total Anomalies Detected by iForest: {np.sum(iso_predictions_mapped)}")

# Optional: See how many REAL attacks it caught
# (This is just for your analysis; in real life, you wouldn't know the true labels for zero-days)
attacks_caught = np.sum((iso_predictions_mapped == 1) & (y_test == 1))
print(f"Real Attacks correctly flagged as Anomalies: {attacks_caught}")