import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# --- STEP 1: LOAD AND PREPROCESS DATA ---

# Load the dataset
df = pd.read_csv(r'C:\College code\Major Project\DataSets\TrafficLabelling\Wednesday-workingHours.pcap_ISCX.csv')

# 1. Clean column names
df.columns = df.columns.str.strip()

# 2. Drop High-Cardinality/Unnecessary Columns (CRITICAL STEP)
# These columns have too many unique values or are just identifiers
drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 'SimillarHTTP']
# Only drop if they actually exist in your dataframe    
existing_drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=existing_drop_cols)

print(f"Dropped columns: {existing_drop_cols}")

# 3. Handle Infinity and NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 4. Define Target (y) and Features (X)
label_col = 'Label'
y = df[label_col].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
X = df.drop(columns=[label_col])

# 5. Encode Remaining Categorical Features
# Now get_dummies will only run on small categories (like Protocol), which is safe.
print("Encoding categorical features...")
X = pd.get_dummies(X, drop_first=True)

# Check the new shape to ensure it's manageable
print(f"New Data Shape: {X.shape}") 

# 6. Scale Features
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# --- STEP 2: SUPERVISED LEARNING ---

# Split dataset
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
print("Applying SMOTE... (this might take a moment)")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest
print("Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_clf.fit(X_train_smote, y_train_smote)

# Predict and Evaluate
rf_predictions = rf_clf.predict(X_test)

print("\n--- Random Forest (Supervised) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(classification_report(y_test, rf_predictions, target_names=['Normal (0)', 'Attack (1)']))