import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Cyber Threat Detector", page_icon="🛡️", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_resources():
    try:
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('iso_forest.pkl', 'rb') as f:
            iso_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
        return rf_model, iso_model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None, None

rf_model, iso_model, scaler, model_columns = load_resources()

# --- HEADER ---
st.title("🛡️ AI-Powered Cyber Threat Detection System")
st.markdown("### Hybrid IDS: Random Forest + Isolation Forest")

if rf_model is None:
    st.error("Error: Model files not found. Please run 'Save_Models.py' first.")
    st.stop()

# --- SIDEBAR: INPUT METHOD ---
st.sidebar.header("Input Data")
input_method = st.sidebar.radio("Choose Input Method:", ["Upload CSV File", "Manual Entry"])

input_df = None

if input_method == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload Network Traffic CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        # Quick clean to match training format
        input_df.columns = input_df.columns.str.strip()
        # Drop ID cols if they exist
        cols_to_drop = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'Timestamp', 'SimillarHTTP', 'Label']
        input_df = input_df.drop(columns=[c for c in cols_to_drop if c in input_df.columns])
        
elif input_method == "Manual Entry":
    st.sidebar.info("Enter values for key features (Simulation)")
    # We create a simplified form for demonstration
    # In a real app, you'd need inputs for all ~70 features, or use defaults
    dest_port = st.sidebar.number_input("Destination Port", min_value=0, max_value=65535, value=80)
    flow_duration = st.sidebar.number_input("Flow Duration", min_value=0, value=1000)
    total_fwd_packets = st.sidebar.number_input("Total Fwd Packets", min_value=0, value=5)
    total_len_fwd = st.sidebar.number_input("Total Length Fwd Packets", min_value=0, value=500)
    
    # Create a DataFrame with 0s for all columns
    data = {col: [0] for col in model_columns}
    input_df = pd.DataFrame(data)
    
    # Fill in the specific values we collected
    # Note: You need to ensure these column names match your CSV exactly
    if 'Destination Port' in input_df.columns: input_df['Destination Port'] = dest_port
    if 'Flow Duration' in input_df.columns: input_df['Flow Duration'] = flow_duration
    if 'Total Fwd Packets' in input_df.columns: input_df['Total Fwd Packets'] = total_fwd_packets
    if 'Total Length of Fwd Packets' in input_df.columns: input_df['Total Length of Fwd Packets'] = total_len_fwd

# --- PREDICTION LOGIC ---
if st.button("Analyze Traffic"):
    if input_df is not None:
        # Align columns: Ensure input has exactly the same columns as training data
        # Missing columns get filled with 0, extra columns are dropped
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Scale
        X_input = scaler.transform(input_df)
        
        # 1. Random Forest Prediction
        rf_pred = rf_model.predict(X_input)
        rf_prob = rf_model.predict_proba(X_input)[:, 1]
        
        # 2. Isolation Forest Prediction
        iso_pred_raw = iso_model.predict(X_input) # -1 is anomaly
        iso_pred = np.where(iso_pred_raw == -1, 1, 0)
        
        # --- DISPLAY RESULTS ---
        
        # Create a dashboard layout for the first row of data
        st.divider()
        st.subheader("Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Logic for Final Status
        # RF says Attack OR (RF says Normal but iForest says Anomaly)
        is_threat = rf_pred[0] == 1
        is_anomaly = iso_pred[0] == 1
        
        with col1:
            st.metric("Supervised Model (RF)", "Attack" if is_threat else "Normal", f"{rf_prob[0]*100:.2f}% Confidence")
            
        with col2:
            st.metric("Unsupervised Model (iForest)", "Anomaly" if is_anomaly else "Standard", " outlier detected" if is_anomaly else "behavior normal")
            
        with col3:
            if is_threat:
                st.error("🚨 HIGH THREAT DETECTED")
            elif is_anomaly:
                st.warning("⚠️ SUSPICIOUS ACTIVITY (Zero-Day Potential)")
            else:
                st.success("✅ TRAFFIC IS SAFE")

        # Visualizing the input data distribution (Optional)
        st.markdown("### Traffic Flow Details")
        st.dataframe(input_df.head())
        
    else:
        st.warning("Please upload a file or enter data to analyze.")