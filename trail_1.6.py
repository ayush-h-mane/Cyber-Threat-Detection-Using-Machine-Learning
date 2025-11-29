# app_final.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import math

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Optional imports for maps/geo - handle gracefully if missing
try:
    from ip2geotools.databases.noncommercial import DbIpCity
    IP2GEO_AVAILABLE = True
except Exception:
    IP2GEO_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False

# ---------------------------
# Page + CSS
# ---------------------------
st.set_page_config(page_title="Cyber Threat Detection — Pro", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0b0d10; color: #e6eef8; }
    .stApp .block-container { padding-top: 1.5rem; }
    .card { background:#0f1116; border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
    .header { text-align:center; font-family: 'Segoe UI', sans-serif; }
    h1 { color: #f5f7fb; }
    .metric-card { background:#121318; padding:12px; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown("<h1 class='header'>🛡️ Cyber Threat Detection — Pro Dashboard</h1>", unsafe_allow_html=True)
st.markdown("A one-file Streamlit app: training, real-time preview, 2D+3D charts, maps, export & more.", unsafe_allow_html=True)

# ---------------------------
# Training function
# ---------------------------
def train_model(df):
    """
    Train RF and Isolation Forest, save rf_model.pkl, iso_forest.pkl, scaler.pkl, model_columns.pkl
    Returns: (success:bool, metrics:dict)
    """
    try:
        df = df.copy()
        df.columns = df.columns.str.strip()

        # preserve original (with possible IP columns) externally if needed
        # Drop high-cardinality and timestamp columns
        drop_cols = ['Flow ID', 'FlowID', 'Source IP', 'Src IP', 'SrcIP', 'Destination IP', 'Dst IP',
                     'DstIP', 'Timestamp', 'SimillarHTTP']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # Clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if 'Label' not in df.columns:
            return False, None

        y = df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        X = df.drop(columns=['Label'])

        # One-hot encode
        X = pd.get_dummies(X, drop_first=True)

        # Save columns for alignment
        model_columns = list(X.columns)
        pickle.dump(model_columns, open("model_columns.pkl", "wb"))

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pickle.dump(scaler, open("scaler.pkl", "wb"))

        # Train-test split (keep pre-smote copy for iso training)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train_pre_smote = X_train.copy()
        y_train_pre_smote = y_train.copy()

        # SMOTE to handle imbalance
        sm = SMOTE(random_state=42)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
        rf.fit(X_train_sm, y_train_sm)
        pickle.dump(rf, open("rf_model.pkl", "wb"))

        # Isolation Forest trained on normal-only in the original X_train (before SMOTE)
        normal_idx = np.where(y_train_pre_smote == 0)[0]
        if len(normal_idx) == 0:
            # fallback: use whole training set (not ideal) but prevents crash
            iso_train_data = X_train_pre_smote
        else:
            iso_train_data = X_train_pre_smote[normal_idx]

        iso = IsolationForest(n_estimators=120, contamination='auto', random_state=42)
        iso.fit(iso_train_data)
        pickle.dump(iso, open("iso_forest.pkl", "wb"))

        # Evaluation metrics on X_test
        y_pred = rf.predict(X_test)
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1 Score": float(f1_score(y_test, y_pred, zero_division=0))
        }

        return True, metrics
    except Exception as e:
        st.error(f"Training error: {e}")
        return False, None

# ---------------------------
# Load models utility
# ---------------------------
@st.cache_resource
def load_models():
    try:
        rf = pickle.load(open("rf_model.pkl", "rb"))
        iso = pickle.load(open("iso_forest.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model_columns = pickle.load(open("model_columns.pkl", "rb"))
        return rf, iso, scaler, model_columns
    except Exception:
        return None, None, None, None

# ---------------------------
# Helper: Clean & align user df for prediction
# ---------------------------
def prepare_input_df(uploaded_df, model_columns):
    df = uploaded_df.copy()
    df.columns = df.columns.str.strip()

    # Keep a copy of original columns (for maps if IP present)
    original_df = df.copy()

    # Drop label if present for prediction
    drop_cols = ['Label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Remove high-cardinality columns used in training pipeline
    drop_more = ['Flow ID', 'FlowID', 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in drop_more if c in df.columns], errors='ignore')

    # One-hot alignment: reindex to model_columns (assumes training saved model_columns from get_dummies)
    df = df.reindex(columns=model_columns, fill_value=0)

    # Clean infinities and NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df, original_df

# ---------------------------
# UI Tabs
# ---------------------------
tab1, tab2 = st.tabs(["🔍 Predict & Visualize", "🧠 Train New Model"])

# ---------------------------
# TAB 1: Predict & Visualize
# ---------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload CSV for Prediction & Live Visualization")
    uploaded_file = st.file_uploader("Upload network traffic CSV (any CICIDS-like CSV works)", type=['csv'])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.markdown("**Preview of uploaded data (first 5 rows)**")
        st.dataframe(raw_df.head())

        # load models
        rf_model, iso_model, scaler, model_columns = load_models()
        if rf_model is None:
            st.warning("No trained model found. Please train a model in the 'Train New Model' tab first.")
        else:
            # Keep a copy with IP columns if present
            input_df, original_df = prepare_input_df(raw_df, model_columns)

            # scale and predict
            try:
                X_scaled = scaler.transform(input_df)
            except Exception as e:
                st.error(f"Scaling error: {e}. Check that your uploaded CSV columns are compatible with the trained model.")
                st.stop()

            rf_pred = rf_model.predict(X_scaled)
            rf_prob = rf_model.predict_proba(X_scaled)[:, 1] if hasattr(rf_model, "predict_proba") else np.zeros(len(rf_pred))
            iso_raw = iso_model.predict(X_scaled)
            iso_pred = np.where(iso_raw == -1, 1, 0)
            final_label = np.array(["Attack" if p == 1 else "Normal" for p in rf_pred])

            # Build results DataFrame
            results = pd.DataFrame({
                "Flow Index": np.arange(1, len(input_df) + 1),
                "RF Prediction": final_label,
                "RF Confidence (%)": (rf_prob * 100).round(2),
                "iForest": ["Anomaly" if x == 1 else "Standard" for x in iso_pred]
            })

            # Top-level KPIs and small dashboard
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].metric("Total Flows", len(results))
            cols[1].metric("Detected Attacks", int((results['RF Prediction'] == 'Attack').sum()))
            cols[2].metric("Suspicious (iForest)", int((results['iForest'] == 'Anomaly').sum()))
            # simple risk score (avg prob)
            avg_prob = float(np.mean(rf_prob)) if len(rf_prob) > 0 else 0.0
            cols[3].metric("Avg Threat Score", f"{avg_prob:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            # 2D Pie chart
            st.markdown("### 📊 Attack vs Normal Distribution")
            pie_df = results['RF Prediction'].value_counts().reset_index()
            pie_df.columns = ['Type', 'Count']
            fig_pie = px.pie(pie_df, names='Type', values='Count',
                             color='Type', color_discrete_map={'Attack': 'red', 'Normal': 'green'},
                             title="Attack vs Normal")
            st.plotly_chart(fig_pie, use_container_width=True)

            # Time-series: threat timeline
            st.markdown("### 📈 Threat Timeline (Flow Index vs Threat Score)")
            ts_df = pd.DataFrame({"FlowIndex": results['Flow Index'], "ThreatScore": rf_prob})
            fig_ts = px.line(ts_df, x='FlowIndex', y='ThreatScore', markers=True, title='Threat Score over Flows')
            st.plotly_chart(fig_ts, use_container_width=True)

            # 3D scatter
            st.markdown("### 🧊 3D Attack Landscape")
            # choose two numeric columns (if available) or use random dims
            def pick_numeric(col_list):
                for c in col_list:
                    if c in input_df.columns and np.issubdtype(input_df[c].dtype, np.number):
                        return c
                return None

            dim_x = pick_numeric(['Flow Duration', 'Duration', 'FlowDuration', 'flow_duration']) or (input_df.columns[0] if len(input_df.columns)>0 else None)
            dim_y = pick_numeric(['Total Fwd Packets', 'Total Fwd Packets ', 'Total Fwd Packets', 'Tot Fwd Pkts']) or (input_df.columns[1] if len(input_df.columns)>1 else None)

            # ensure x,y present; if not create synthetic ones
            x_vals = input_df[dim_x] if dim_x in input_df.columns else np.random.randn(len(input_df))
            y_vals = input_df[dim_y] if dim_y in input_df.columns else np.random.randn(len(input_df))

            df_3d = pd.DataFrame({
                "x": x_vals,
                "y": y_vals,
                "z": rf_prob,
                "Threat": final_label
            })
            fig3d = px.scatter_3d(df_3d, x='x', y='y', z='z', color='Threat',
                                  color_discrete_map={'Attack': 'red', 'Normal': 'green'},
                                  title="3D Attack Landscape", height=600)
            st.plotly_chart(fig3d, use_container_width=True)

            # Pagination + color-coded table + expanders
            st.markdown("### 🧾 Prediction Table (Paginated, color-coded)")
            rows_per_page = st.slider("Rows per page:", 5, 200, 20, key="rows_per_page")
            total_pages = max(1, math.ceil(len(results) / rows_per_page))
            page_num = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1, key="page_num")
            start = (page_num - 1) * rows_per_page
            end = min(start + rows_per_page, len(results))

            display_df = results.iloc[start:end].reset_index(drop=True)

            # color mapping
            def color_row(r):
                return ['background-color: #ffdddd' if r['RF Prediction'] == 'Attack' else 'background-color: #ddffdd'] * len(r)

            st.dataframe(display_df.style.apply(color_row, axis=1), use_container_width=True)

            # Expandable per flow with feature snapshot
            st.markdown("### 🔎 Flow Detail (Expand rows to view feature values)")
            for idx in range(start, end):
                with st.expander(f"Flow {int(results['Flow Index'].iloc[idx-start])} — {results['RF Prediction'].iloc[idx-start]}"):
                    c1, c2 = st.columns([2,3])
                    c1.metric("RF Prediction", results['RF Prediction'].iloc[idx-start])
                    c1.metric("Confidence", f"{results['RF Confidence (%)'].iloc[idx-start]}%")
                    c1.metric("iForest", results['iForest'].iloc[idx-start])
                    # show the feature values (original aligned input)
                    c2.write("Input feature snapshot:")
                    st.dataframe(input_df.iloc[idx:idx+1].T)

            # Export predictions
            csv_bytes = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            # -----------------------
            # Map visualization (IP -> geo)
            # -----------------------
            st.markdown("### 🌍 Attack Map (IP Geolocation)")
            # Try to find IP column in original_df (common names)
            ip_cols_candidates = [c for c in original_df.columns if 'ip' in c.lower() or 'src' in c.lower() and 'ip' in c.lower() or 'dst' in c.lower() and 'ip' in c.lower()]
            # brute check common names:
            fallback_candidates = ['Source IP', 'Src IP', 'Destination IP', 'Dst IP', 'SrcIP', 'DstIP']
            ip_col = None
            for c in fallback_candidates:
                if c in original_df.columns:
                    ip_col = c
                    break
            # if still none, search generically for columns that look like ip addresses
            if ip_col is None:
                for c in original_df.columns:
                    sample = original_df[c].astype(str).dropna().head(10).tolist()
                    if all(isinstance(s, str) and s.count('.')>=1 and all(p.isdigit() for p in s.split('.') if p.isdigit() or True) for s in sample if s):
                        ip_col = c
                        break

            if ip_col is not None:
                st.write(f"Using IP column: **{ip_col}** (showing up to 200 points)")
                # build coordinates
                coords = []
                limit = min(200, len(original_df))
                for i, ip in enumerate(original_df[ip_col].astype(str).head(limit)):
                    lat = None; lon = None
                    if IP2GEO_AVAILABLE:
                        try:
                            res = DbIpCity.get(ip, api_key="free")
                            lat, lon = res.latitude, res.longitude
                        except Exception:
                            lat, lon = None, None
                    if lat is None or lon is None:
                        # fallback: random global coordinate (so map shows something) - seeded for deterministic results
                        rng = np.random.RandomState(hash(str(ip)) % 2**32)
                        lat = float(rng.uniform(-60, 70))
                        lon = float(rng.uniform(-180, 180))

                    coords.append({"lat": lat, "lon": lon, "ip": ip, "index": i+1, "prediction": final_label[i] if i < len(final_label) else "Unknown"})

                map_df = pd.DataFrame(coords)

                if PYDECK_AVAILABLE:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position='[lon, lat]',
                        get_fill_color='[255, 0, 0, 160]',
                        get_radius=100000,
                        pickable=True
                    )
                    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=30)
                    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/dark-v10')
                    st.pydeck_chart(deck)
                else:
                    st.map(map_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'}).loc[:, ['latitude', 'longitude']])
            else:
                st.info("No IP-like column found in uploaded file — to show map, include 'Source IP' or similar in CSV.")

            # -----------------------
            # Real-time UI: toggle
            # -----------------------
            st.markdown("### ⚡ Real-Time Monitor (Simulated Live)")
            realtime_toggle = st.checkbox("Enable Real-Time Mode (auto-refresh)", value=False)

            if realtime_toggle:
                placeholder = st.empty()
                # run for a limited number of updates to avoid infinite loop
                updates = st.slider("How many live updates to run?", 5, 200, 20, key="live_updates")
                delay = st.slider("Update interval (seconds)", 1, 10, 2, key="live_delay")
                for i in range(updates):
                    with placeholder.container():
                        current_index = min(len(results)-1, i % len(results))
                        attacks = int((results['RF Prediction'] == 'Attack').sum())
                        normals = int((results['RF Prediction'] == 'Normal').sum())

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Live update", f"{i+1}/{updates}")
                        c2.metric("Attacks", attacks)
                        c3.metric("Normals", normals)

                        # live sparkline of last N scores
                        last_n = min(200, len(rf_prob))
                        spark = pd.DataFrame({"idx": list(range(last_n)), "score": rf_prob[:last_n]})
                        fig_live = px.line(spark, x='idx', y='score', title="Live Threat Spark")
                        st.plotly_chart(fig_live, use_container_width=True)

                        # small detailed card of a rotating flow
                        st.markdown("**Spotlight Flow**")
                        st.write(f"- Index: {current_index+1}")
                        st.write(f"- Prediction: {results['RF Prediction'].iloc[current_index]}")
                        st.write(f"- Confidence: {results['RF Confidence (%)'].iloc[current_index]}%")
                    time.sleep(delay)

            # Animated threat timeline
            st.markdown("### 🎬 Animated Threat Timeline (Threat Score Animation)")
            anim_df = pd.DataFrame({"Flow": list(range(len(rf_prob))), "ThreatScore": rf_prob})
            try:
                fig_anim = px.scatter(anim_df, x='Flow', y='ThreatScore', animation_frame='Flow',
                                      range_y=[0,1], color='ThreatScore', color_continuous_scale='Reds',
                                      title="Threat Score Animation")
                st.plotly_chart(fig_anim, use_container_width=True)
            except Exception as e:
                st.info("Animation not available due to browser limitations.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# TAB 2: Train New Model
# ---------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload CSV to Train New Model (creates new model files)")
    train_file = st.file_uploader("Training CSV (must include 'Label' column)", type=['csv'], key="train_file")

    if train_file is not None:
        train_df = pd.read_csv(train_file)
        st.write("Training preview:")
        st.dataframe(train_df.head())

        if st.button("Start Training"):
            with st.spinner("Training the models — this may take 20–90 seconds depending on data & CPU..."):
                success, metrics = train_model(train_df)
            if success:
                st.success("Training complete! Models saved (rf_model.pkl, iso_forest.pkl, scaler.pkl, model_columns.pkl).")
                if metrics:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
                    c2.metric("Precision", f"{metrics['Precision']:.3f}")
                    c3.metric("Recall", f"{metrics['Recall']:.3f}")
                    c4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
            else:
                st.error("Training failed. Ensure 'Label' exists and data is clean.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer / help
# ---------------------------
st.markdown("---")
st.markdown("Need further customization? I can add: filters, colored table cells, alert thresholds, or a live socket input from a packet-capture stream. Just ask!")
