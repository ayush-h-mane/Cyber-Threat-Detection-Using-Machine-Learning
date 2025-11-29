import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# -------------------------------------------------------------
# STREAMLIT PAGE SETUP + CUSTOM UI
# -------------------------------------------------------------
st.set_page_config(page_title="Cyber Threat Detection", page_icon="🛡️", layout="wide")

# Custom CSS for Stylish UI
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: white;
        font-family: 'Segoe UI';
    }
    .block-container {
        padding-top: 2rem;
    }
    .card {
        background: #1e1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px #00000055;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #22232b;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border-left: 6px solid #4CAF50;
    }
    .metric-title {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🛡️ AI-Powered Cyber Threat Detection System</h1>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------------
def train_model(df):
    df.columns = df.columns.str.strip()

    drop_cols = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP', 'Dst IP',
                 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if "Label" not in df.columns:
        st.error("Dataset MUST contain a 'Label' column.")
        return False, None

    y = df['Label'].apply(lambda x: 0 if str(x).strip() == "BENIGN" else 1)
    X = df.drop(columns=['Label'])
    X = pd.get_dummies(X, drop_first=True)

    model_columns = list(X.columns)
    pickle.dump(model_columns, open("model_columns.pkl", "wb"))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pickle.dump(rf, open("rf_model.pkl", "wb"))

    normal_only = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=120, contamination='auto', random_state=42)
    iso.fit(normal_only)
    pickle.dump(iso, open("iso_forest.pkl", "wb"))

    # Compute training accuracy
    train_pred = rf.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, train_pred),
        "Precision": precision_score(y_test, train_pred),
        "Recall": recall_score(y_test, train_pred),
        "F1 Score": f1_score(y_test, train_pred),
    }

    return True, metrics


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
    st.subheader("Upload CSV for Prediction")

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

            df = df.reindex(columns=model_columns, fill_value=0)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            X_input = scaler.transform(df)

            if st.button("Analyze Traffic"):
                rf_pred = rf_model.predict(X_input)
                rf_prob = rf_model.predict_proba(X_input)[:, 1]
                iso_pred = np.where(iso_model.predict(X_input) == -1, 1, 0)

                final_results = ["Attack" if p == 1 else "Normal" for p in rf_pred]

                # 🎨 PIE CHART
                st.markdown("### 📊 Attack vs Normal Distribution")
                pie_df = pd.DataFrame({
                    "Type": ["Attack", "Normal"],
                    "Count": [sum(rf_pred), len(rf_pred) - sum(rf_pred)]
                })
                fig_pie = px.pie(pie_df, names='Type', values='Count',
                                 color='Type',
                                 color_discrete_map={"Attack": "red", "Normal": "green"})
                st.plotly_chart(fig_pie, use_container_width=True)

                # 🎨 TIME SERIES CHART
                st.markdown("### 📈 Threat Timeline (Flow Index vs Prediction)")
                ts_df = pd.DataFrame({
                    "Flow": range(len(df)),
                    "Threat": rf_pred
                })
                fig_ts = px.line(ts_df, x="Flow", y="Threat",
                                 markers=True,
                                 color_discrete_sequence=["#ff4b4b"])
                fig_ts.update_yaxes(title="Threat (1 = Attack)")
                st.plotly_chart(fig_ts, use_container_width=True)

                # PAGINATION
                rows_per_page = st.slider("Rows per page:", 5, 100, 20)
                page_num = st.number_input("Page:", 1, (len(df)-1)//rows_per_page+1)
                start = (page_num - 1) * rows_per_page
                end = min(start + rows_per_page, len(df))

                # Table results
                results = pd.DataFrame({
                    "Flow ID": range(1, len(df) + 1),
                    "RF Prediction": final_results,
                    "Confidence (%)": (rf_prob * 100).round(2),
                    "iForest": ["Anomaly" if x else "Standard" for x in iso_pred]
                })

                st.markdown("### 🧾 Prediction Table")
                st.dataframe(results.iloc[start:end])

                # Expandable detail panels
                st.markdown("### 🔎 Detailed Flow Breakdown")
                for i in range(start, end):
                    with st.expander(f"Flow {i+1} — {final_results[i]}"):
                        col1, col2 = st.columns(2)
                        col1.metric("Prediction", final_results[i])
                        col2.metric("Confidence", f"{rf_prob[i]*100:.2f}%")

                        st.write("### Feature Values")
                        st.dataframe(df.iloc[i:i+1].T)

                # Export Button
                st.download_button(
                    "📥 Download Predictions as CSV",
                    data=results.to_csv(index=False),
                    file_name="cyber_predictions.csv",
                    mime="text/csv"
                )


# =============================================================
# TAB 2 — TRAIN MODEL
# =============================================================
with tab_train:
    st.subheader("Train a New Model")

    train_file = st.file_uploader("Upload Training Dataset", type=['csv'], key="train_dataset")

    if train_file:
        df_train = pd.read_csv(train_file)
        st.write("Preview:", df_train.head())

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                success, metrics = train_model(df_train)

            if success:
                st.success("🎉 Training complete!")

                st.markdown("### 📊 Training Metrics")
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
                colB.metric("Precision", f"{metrics['Precision']:.2f}")
                colC.metric("Recall", f"{metrics['Recall']:.2f}")
                colD.metric("F1 Score", f"{metrics['F1 Score']:.2f}")
