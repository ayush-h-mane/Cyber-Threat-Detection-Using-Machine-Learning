import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------
# STREAMLIT SETUP
# -------------------------------------------------------------
st.set_page_config(page_title="Cyber Threat Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-Powered Cyber Threat Detection System")

# -------------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------------
def train_model(df):
    df.columns = df.columns.str.strip()

    # Drop unnecessary columns
    drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
                 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle NaN & INF
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Must have Label column
    if "Label" not in df.columns:
        st.error("Dataset MUST contain a 'Label' column.")
        return False

    # Encode target
    y = df['Label'].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
    X = df.drop(columns=['Label'])

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Save model columns
    model_columns = list(X.columns)
    pickle.dump(model_columns, open("model_columns.pkl", "wb"))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    rf.fit(X_train_smote, y_train_smote)
    pickle.dump(rf, open("rf_model.pkl", "wb"))

    # Train Isolation Forest
    normal_only = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=80, contamination='auto', random_state=42)
    iso.fit(normal_only)
    pickle.dump(iso, open("iso_forest.pkl", "wb"))

    return True


# -------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        rf = pickle.load(open("rf_model.pkl", "rb"))
        iso = pickle.load(open("iso_forest.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model_columns = pickle.load(open("model_columns.pkl", "rb"))
        return rf, iso, scaler, model_columns
    except:
        return None, None, None, None


# -------------------------------------------------------------
# UI TABS
# -------------------------------------------------------------
tab_predict, tab_train = st.tabs(["🔍 Predict Traffic", "🧠 Train New Model"])

# =============================================================
# TAB 1 — PREDICT
# =============================================================
with tab_predict:
    st.header("Upload CSV for Prediction")

    rf_model, iso_model, scaler, model_columns = load_models()

    if rf_model is None:
        st.warning("⚠️ No trained model found. Train a model in the next tab.")
    else:
        uploaded = st.file_uploader("Upload Network Traffic CSV", type=['csv'])

        if uploaded:
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()

            drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
                         'Timestamp', 'SimillarHTTP', 'Label']

            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

            # Align with training columns
            df = df.reindex(columns=model_columns, fill_value=0)

            # 🔥 FIX — REMOVE Infinity & NaN (this prevented your error)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # Now scale safely
            X_input = scaler.transform(df)

            if st.button("Analyze Traffic"):
                rf_pred = rf_model.predict(X_input)
                rf_prob = rf_model.predict_proba(X_input)[:, 1]
                iso_pred = np.where(iso_model.predict(X_input) == -1, 1, 0)

                st.success("Analysis Completed!")
                st.dataframe(df.head())

                st.subheader("Results")

                for i in range(len(df)):
                    result = "🚨 Attack" if rf_pred[i] == 1 else "✅ Normal"
                    st.write(f"Flow {i+1}: {result}")


# =============================================================
# TAB 2 — TRAIN NEW MODEL
# =============================================================
with tab_train:
    st.header("Upload CSV to Train New Model")

    train_file = st.file_uploader("Upload Training Dataset", type=['csv'], key="train_file")

    if train_file:
        df_train = pd.read_csv(train_file)
        st.write("Preview:", df_train.head())

        if st.button("Train Model"):
            with st.spinner("Training... Please wait (20–60 seconds)..."):
                success = train_model(df_train)

            if success:
                st.success("🎉 Training complete! Models have been updated.")
            else:
                st.error("Training failed. Please check dataset format.")
