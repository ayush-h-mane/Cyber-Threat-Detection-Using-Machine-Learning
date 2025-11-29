import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Cyber Threat Detector", page_icon="🛡️", layout="wide")

# --------------------------------------------------------
# FUNCTION: TRAIN NEW MODEL
# --------------------------------------------------------
def train_model(df):
    df.columns = df.columns.str.strip()

    # Drop problematic columns
    cols_to_drop = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 
                    'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Handle missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode target
    if "Label" not in df.columns:
        st.error("Dataset must contain 'Label' column.")
        return None

    y = df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    X = df.drop(columns=['Label'])

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Save column order
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    rf.fit(X_train_smote, y_train_smote)

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)

    # Train Isolation Forest
    normal_only = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=80, contamination='auto', random_state=42)
    iso.fit(normal_only)

    with open('iso_forest.pkl', 'wb') as f:
        pickle.dump(iso, f)

    return True


# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        rf = pickle.load(open('rf_model.pkl', 'rb'))
        iso = pickle.load(open('iso_forest.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        cols = pickle.load(open('model_columns.pkl', 'rb'))
        return rf, iso, scaler, cols
    except:
        return None, None, None, None


# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
st.title("🛡️ AI-Powered Cyber Threat Detection System")

tab1, tab2 = st.tabs(["🔍 Predict Traffic", "🧠 Train New Model"])

# ========================================================
# TAB 1 — PREDICTION
# ========================================================
with tab1:
    st.header("Upload CSV for Prediction")

    rf_model, iso_model, scaler, model_columns = load_models()

    if rf_model is None:
        st.warning("No trained model found. Please train a model in the next tab.")
    else:
        uploaded = st.file_uploader("Upload Traffic CSV", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns
    drop_cols = ['Flow ID', 'Source IP', 'Src IP',
                 'Destination IP', 'Dst IP', 'Timestamp',
                 'SimillarHTTP', 'Label']

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Align columns to training model
    df = df.reindex(columns=model_columns, fill_value=0)

    # 🔥 FIX: Remove infinities and NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)    

    # FINAL: Scale safely
    X = scaler.transform(df)

    if st.button("Analyze"):
        rf_pred = rf_model.predict(X)
        rf_prob = rf_model.predict_proba(X)[:, 1]
        iso_pred = np.where(iso_model.predict(X) == -1, 1, 0)

        st.success("Analysis Complete")
        st.write(df.head())

        for i in range(len(df)):
            st.write(
                f"Flow {i+1}:",
                "🚨 Attack" if rf_pred[i] == 1 else "✅ Normal"
            )


# ========================================================
# TAB 2 — TRAINING
# ========================================================
with tab2:
    st.header("Train a New Model with Any CSV File")

    train_file = st.file_uploader("Upload Training Dataset", type=['csv'])

    if train_file:
        df_train = pd.read_csv(train_file)
        st.write(df_train.head())

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                ok = train_model(df_train)

            if ok:
                st.success("🎉 Training complete! Models updated.")
            else:
                st.error("Training failed. Check dataset format.")
