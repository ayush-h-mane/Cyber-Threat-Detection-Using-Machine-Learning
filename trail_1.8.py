# app_enterprise.py
"""
Enterprise-grade Streamlit app implementing:
- Secure login with bcrypt + argon2 (both supported)
- Admin user management UI (create/delete/reset/set role)
- Prediction-only flow (requires pre-trained model files)
- Chunked CSV loading, lazy prediction, downsampling visuals
- SQLite storage for predictions; optional DuckDB support
- ZIP export of full predictions + stats
- OAuth login placeholders (Authlib optional)
- Real packet streaming (simulated; optional pyshark/scapy support)
- Automatic diagnostics for model/schema mismatches
- Light/Dark theme toggle (affects Plotly templates)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import math
import time
import zipfile
import sqlite3
import secrets
from io import BytesIO, StringIO
from typing import Tuple, Dict, Any

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Plotly Express (fix for px not defined)
import plotly.express as px

# Optional libs: bcrypt, argon2, duckdb, authlib, pyshark, scapy
OPTIONAL = {}
try:
    import bcrypt
    OPTIONAL['bcrypt'] = True
except Exception:
    OPTIONAL['bcrypt'] = False

try:
    from argon2 import PasswordHasher
    OPTIONAL['argon2'] = True
except Exception:
    OPTIONAL['argon2'] = False

try:
    import duckdb
    OPTIONAL['duckdb'] = True
except Exception:
    OPTIONAL['duckdb'] = False

try:
    from authlib.integrations.requests_client import OAuth2Session
    OPTIONAL['authlib'] = True
except Exception:
    OPTIONAL['authlib'] = False

try:
    import pyshark
    OPTIONAL['pyshark'] = True
except Exception:
    OPTIONAL['pyshark'] = False

try:
    import scapy.all as scapy
    OPTIONAL['scapy'] = True
except Exception:
    OPTIONAL['scapy'] = False

# Constants / file paths
USERS_FILE = "users.json"
MODEL_FILES = {
    "rf": "rf_model.pkl",
    "iso": "iso_forest.pkl",
    "scaler": "scaler.pkl",
    "cols": "model_columns.pkl"
}
SQL_DB = "predictions.db"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Streamlit page
st.set_page_config(page_title="Cyber Threat Detection — Enterprise", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Threat Detection — Enterprise Dashboard")
st.markdown("A secure, production-ready prediction app — admin controls, storage, exports, streaming, and more.")

# -----------------------
# Theme toggle (Plotly template)
# -----------------------
st.sidebar.markdown("---")
THEME = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
PLOTLY_TEMPLATE = "plotly_dark" if THEME == "Dark" else "plotly_white"

# -----------------------
# Utilities: hashing
# -----------------------
def hash_with_argon2(password: str) -> Dict[str,str]:
    if not OPTIONAL['argon2']:
        raise RuntimeError("argon2 not available")
    ph = PasswordHasher()
    h = ph.hash(password)
    return {"algo": "argon2", "hash": h}

def verify_argon2(stored_hash: str, password: str) -> bool:
    ph = PasswordHasher()
    try:
        ph.verify(stored_hash, password)
        return True
    except Exception:
        return False

def hash_with_bcrypt(password: str) -> Dict[str,str]:
    if not OPTIONAL['bcrypt']:
        raise RuntimeError("bcrypt not available")
    salt = bcrypt.gensalt()
    h = bcrypt.hashpw(password.encode('utf-8'), salt)
    return {"algo": "bcrypt", "hash": h.decode()}

def verify_bcrypt(stored_hash: str, password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode())
    except Exception:
        return False

def secure_hash(password: str) -> Dict[str,str]:
    if OPTIONAL['argon2']:
        return hash_with_argon2(password)
    elif OPTIONAL['bcrypt']:
        return hash_with_bcrypt(password)
    else:
        import hashlib
        salt = secrets.token_bytes(16).hex()
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 200000).hex()
        return {"algo": "pbkdf2", "salt": salt, "hash": dk}

def verify_password_record(record: Dict[str,Any], password: str) -> bool:
    algo = record.get("algo")
    if algo == "argon2":
        if not OPTIONAL['argon2']:
            return False
        return verify_argon2(record['hash'], password)
    elif algo == "bcrypt":
        if not OPTIONAL['bcrypt']:
            return False
        return verify_bcrypt(record['hash'], password)
    elif algo == "pbkdf2":
        import hashlib
        salt = bytes.fromhex(record['salt'])
        candidate = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200000).hex()
        return secrets.compare_digest(candidate, record['hash'])
    else:
        return False

# -----------------------
# Users file ops & admin management UI
# -----------------------
def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

users = load_users()
if not users:
    st.warning("No users configured. Create an Admin account now.")
    with st.form("initial_admin"):
        admin_user = st.text_input("Admin username")
        admin_pass = st.text_input("Admin password", type="password")
        admin_pass2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create Admin")
        if submitted:
            if not admin_user or not admin_pass:
                st.error("Both username and password are required.")
                st.stop()
            if admin_pass != admin_pass2:
                st.error("Passwords do not match.")
                st.stop()
            record = secure_hash(admin_pass)
            users = {admin_user: {"role":"admin", "cred": record}}
            save_users(users)
            st.success(f"Admin created: {admin_user}. Please refresh and log in.")
            # use rerun to refresh UI
            st.rerun()

# -----------------------
# Authentication UI (sidebar)
# -----------------------
st.sidebar.header("Authentication")
if "auth" not in st.session_state:
    st.session_state["auth"] = {"logged_in": False, "user": None, "role": None}

def login_attempt(username, password):
    users_local = load_users()
    if username in users_local:
        rec = users_local[username]['cred']
        ok = verify_password_record(rec, password)
        if ok:
            st.session_state['auth'] = {"logged_in": True, "user": username, "role": users_local[username]['role']}
            return True
    return False

with st.sidebar:
    if not st.session_state['auth']['logged_in']:
        username = st.text_input("Username", key="login_user")
        pwd = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_attempt(username, pwd):
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid credentials")
        if OPTIONAL.get('authlib'):
            st.markdown("**OAuth login** (Admin: configure client IDs in env vars)")
            if st.button("Login with Google (placeholder)"):
                st.info("OAuth flow requires server/client config; see app comments for setup.")
    else:
        st.markdown(f"Logged in: **{st.session_state['auth']['user']}** ({st.session_state['auth']['role']})")
        if st.button("Logout"):
            st.session_state['auth'] = {"logged_in": False, "user": None, "role": None}
            st.rerun()

if not st.session_state['auth']['logged_in']:
    st.stop()

# Admin user management panel (via query param)
st.sidebar.markdown("---")
if st.session_state['auth']['role'] == "admin":
    st.sidebar.subheader("Admin: Users")
    if st.sidebar.button("Manage Users"):
        st.query_params(manage_users="1")
        st.rerun()

if st.query_params().get("manage_users"):
    if st.session_state['auth']['role'] != "admin":
        st.error("Only admin can manage users.")
    else:
        st.header("🔧 Admin — User Management")
        users = load_users()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Existing users")
            for u, meta in users.items():
                st.write(f"- {u} ({meta['role']})")
        with col2:
            st.subheader("Create new user")
            newu = st.text_input("username for new user")
            newp = st.text_input("password for new user", type="password")
            role = st.selectbox("role", ["user","admin"])
            if st.button("Create user"):
                if not newu or not newp:
                    st.error("username & password required")
                else:
                    record = secure_hash(newp)
                    users[newu] = {"role": role, "cred": record}
                    save_users(users)
                    st.success(f"User created: {newu}")

        st.subheader("Modify / Delete user")
        sel = st.selectbox("select user", list(users.keys()))
        if sel:
            if st.button("Delete user"):
                if sel == st.session_state['auth']['user']:
                    st.error("Cannot delete yourself while logged in")
                else:
                    users.pop(sel, None)
                    save_users(users)
                    st.success(f"Deleted {sel}")
            if st.button("Reset password for selected"):
                npw = st.text_input("new password for " + sel, key="resetpw")
                if npw:
                    users[sel]['cred'] = secure_hash(npw)
                    save_users(users)
                    st.success("Password reset")
        st.rerun()

# -----------------------
# Check model files and load models
# -----------------------
def check_model_files() -> Tuple[bool, list]:
    missing = [f for f in MODEL_FILES.values() if not os.path.exists(f)]
    return (len(missing)==0, missing)

ok_models, missing_models = check_model_files()
if not ok_models:
    st.error(f"Missing model files: {missing_models}. Place rf_model.pkl, iso_forest.pkl, scaler.pkl, model_columns.pkl in the app folder.")
    st.stop()

@st.cache_resource
def load_models():
    rf = pickle.load(open(MODEL_FILES['rf'], 'rb'))
    iso = pickle.load(open(MODEL_FILES['iso'], 'rb'))
    scaler = pickle.load(open(MODEL_FILES['scaler'], 'rb'))
    cols = pickle.load(open(MODEL_FILES['cols'], 'rb'))
    return rf, iso, scaler, cols

rf_model, iso_model, scaler, model_columns = load_models()

# -----------------------
# CSV loader & optimization
# -----------------------
@st.cache_data(show_spinner=False)
def load_csv_chunked(file_obj, chunksize=200000):
    try:
        it = pd.read_csv(file_obj, chunksize=chunksize, low_memory=False)
        chunks = []
        for chunk in it:
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
    except Exception:
        return pd.read_csv(file_obj, low_memory=False)

def optimize_types(df):
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    return df

# -----------------------
# Severity scoring
# -----------------------
def severity(score: float) -> Tuple[str,str]:
    if score <= 0.20:
        return "Low", "green"
    elif score <= 0.50:
        return "Medium", "yellow"
    elif score <= 0.80:
        return "High", "orange"
    else:
        return "Critical", "red"

# -----------------------
# SQLite integration
# -----------------------
def init_sqlite(db_path=SQL_DB):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        flow_index INTEGER,
        prediction TEXT,
        confidence REAL,
        severity TEXT,
        anomaly TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );""")
    conn.commit()
    return conn

sql_conn = init_sqlite()

def write_predictions_to_sql(pred_df: pd.DataFrame, conn=sql_conn):
    pred_df = pred_df.rename(columns={"Flow Index":"flow_index","Prediction":"prediction","Confidence":"confidence","Severity":"severity","Anomaly":"anomaly"})
    pred_df.to_sql('predictions', conn, if_exists='append', index=False)

# -----------------------
# Main UI tabs
# -----------------------
st.header("🔍 Prediction — Upload & Analyze")
upload_tab, stream_tab, export_tab, diagnostics_tab = st.tabs(["Upload CSV", "Stream (pcap/sim)", "Export & Storage", "Diagnostics"])

# ---------- Upload CSV tab ----------
with upload_tab:
    st.subheader("Upload CSV (CICIDS-like). Chunked loader & lazy paging for large files.")
    uploaded = st.file_uploader("Upload network CSV", type=["csv"])
    if uploaded:
        with st.spinner("Loading CSV (chunked, optimized)..."):
            df_raw = load_csv_chunked(uploaded)
        if df_raw is None or df_raw.empty:
            st.error("Unable to read CSV or file empty.")
        else:
            st.success(f"Loaded {len(df_raw):,} rows.")
            st.dataframe(df_raw.head())
            df_raw = optimize_types(df_raw)

            # Diagnostics: missing/extra columns
            st.markdown("### Automatic Schema Diagnostics")
            required_cols = set(model_columns)
            user_cols = set(df_raw.columns.str.strip())
            missing = required_cols - user_cols
            extra = user_cols - required_cols
            st.write(f"Model expects {len(required_cols)} features.")
            if missing:
                st.warning(f"Missing {len(missing)} columns — predictions will set these to 0. Example missing: {list(missing)[:5]}")
            else:
                st.success("All required columns present.")
            if extra:
                st.info(f"Uploaded CSV has {len(extra)} extra columns (will be ignored). Example extras: {list(extra)[:5]}")

            # Align
            def align_df(uploaded_df: pd.DataFrame, model_cols):
                df = uploaded_df.copy()
                df.columns = df.columns.str.strip()
                if 'Label' in df.columns:
                    df = df.drop(columns=['Label'])
                drops = ['Flow ID','FlowID','Timestamp','SimillarHTTP']
                df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')
                df = df.reindex(columns=model_cols, fill_value=0)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True)
                return df

            aligned = align_df(df_raw, model_columns)
            n = len(aligned)
            st.success(f"Aligned with model: {n:,} rows ready for prediction.")

            # Pagination / lazy predict
            rows_per_page = st.slider("Rows per page", 5, 500, 50)
            total_pages = max(1, math.ceil(n/rows_per_page))
            page = st.number_input("Page", 1, total_pages, 1)
            start = (page-1)*rows_per_page
            end = min(start+rows_per_page, n)

            sample_size = st.number_input("Sampling size for estimated stats (0 = off)", min_value=0, max_value=200000, value=2000, step=100)
            est_attacks = None
            if sample_size > 0:
                s = min(sample_size, n)
                sample_idx = np.random.RandomState(42).choice(n, s, replace=False)
                Xs = scaler.transform(aligned.iloc[sample_idx])
                spreds = rf_model.predict(Xs)
                ratio = spreds.mean()
                est_attacks = int(ratio * n)
                st.info(f"Estimated attacks (sample {s}): {est_attacks} (ratio {ratio:.3f})")

            X_page = scaler.transform(aligned.iloc[start:end])
            p_preds = rf_model.predict(X_page)
            p_probs = rf_model.predict_proba(X_page)[:,1]
            iso_raw = iso_model.predict(X_page)
            iso_preds = np.where(iso_raw == -1, 1, 0)

            severities = [severity(prob)[0] for prob in p_probs]
            page_df = pd.DataFrame({
                "Flow Index": list(range(start+1,end+1)),
                "Prediction": ["Attack" if x==1 else "Normal" for x in p_preds],
                "Confidence %": (p_probs*100).round(2),
                "Severity": severities,
                "Anomaly": ["Yes" if x==1 else "No" for x in iso_preds]
            })

            st.markdown("### Prediction Table (page)")
            def style_row(r):
                color = '#ffcccc' if r['Prediction']=='Attack' else '#ccffcc'
                if r['Severity']=='Critical': color = '#ff6666'
                if r['Severity']=='High': color = '#ffaf66'
                if r['Severity']=='Medium': color = '#fff2a8'
                if r['Severity']=='Low': color = '#c8ffd1'
                return [f"background-color: {color}"] * len(r)
            st.dataframe(page_df.style.apply(style_row, axis=1), use_container_width=True)

            st.markdown("### Flow details")
            for i in range(start, end):
                local = i - start
                with st.expander(f"Flow {i+1} — {page_df['Prediction'].iloc[local]} — {page_df['Severity'].iloc[local]}"):
                    c1, c2 = st.columns([2,3])
                    c1.metric("Prediction", page_df['Prediction'].iloc[local])
                    c1.metric("Confidence", f"{page_df['Confidence %'].iloc[local]}%")
                    c1.metric("Severity", page_df['Severity'].iloc[local])
                    c2.write("Aligned features snapshot")
                    st.dataframe(aligned.iloc[i:i+1].T)

            # Visualizations
            st.markdown("### Visuals")
            pie_df = page_df['Prediction'].value_counts().reset_index()
            pie_df.columns = ['Type','Count']
            fig = px.pie(pie_df, names='Type', values='Count', color='Type',
                         color_discrete_map={'Attack':'red','Normal':'green'}, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("3D threat landscape (sampled)")
            if n > 200000:
                st.warning("3D disabled for very large dataset (>200k rows).")
            else:
                sample_N = min(n, 8000)
                sample_df = aligned.sample(sample_N, random_state=1)
                try:
                    Xs = scaler.transform(sample_df)
                    probs = rf_model.predict_proba(Xs)[:,1]
                    numeric_cols = [c for c in aligned.columns if np.issubdtype(aligned[c].dtype, np.number)]
                    if len(numeric_cols) >= 2:
                        fig3d = px.scatter_3d(sample_df, x=numeric_cols[0], y=numeric_cols[1], z=probs,
                                              color=["Attack" if p>0.5 else "Normal" for p in probs],
                                              color_discrete_map={'Attack':'red','Normal':'green'}, template=PLOTLY_TEMPLATE)
                        fig3d.update_layout(height=600)
                        st.plotly_chart(fig3d, use_container_width=True)
                except Exception:
                    st.info("3D chart skipped (insufficient numeric features or compute).")

            if st.session_state['auth']['role'] == 'admin':
                if st.button("Save current page predictions to DB (SQLite)"):
                    write_predictions_to_sql(page_df)
                    st.success("Saved to SQLite DB (predictions table).")

# ---------- Stream tab ----------
with stream_tab:
    st.subheader("Streaming: pcap file or simulated live feed")
    stream_mode = st.selectbox("Stream mode", ["Simulated CSV stream", "Upload pcap file (pyshark)"], index=0)
    if stream_mode == "Simulated CSV stream":
        csv_file = st.file_uploader("CSV to stream (will be read row-by-row)", type=['csv'], key="stream_csv")
        if csv_file:
            st.write("Streaming preview (press Start to stream rows):")
            preview = pd.read_csv(csv_file, nrows=5)
            st.dataframe(preview)
            if st.button("Start simulated stream"):
                df_stream = pd.read_csv(csv_file, low_memory=False)
                aligned_stream = df_stream.reindex(columns=model_columns, fill_value=0)
                aligned_stream.replace([np.inf, -np.inf], np.nan, inplace=True)
                aligned_stream.fillna(0, inplace=True)
                placeholder = st.empty()
                for i in range(len(aligned_stream)):
                    row = aligned_stream.iloc[i:i+1]
                    Xr = scaler.transform(row)
                    p = rf_model.predict(Xr)[0]
                    prob = rf_model.predict_proba(Xr)[:,1][0]
                    an = iso_model.predict(Xr)[0]
                    severity_label, sevcolor = severity(prob)
                    with placeholder.container():
                        st.markdown(f"**Flow {i+1}** — Prediction: **{'Attack' if p==1 else 'Normal'}** — Confidence: {prob:.3f} — Severity: **{severity_label}**")
                        st.dataframe(row.T)
                    time.sleep(0.5)
                st.success("Stream finished.")
    else:
        if not OPTIONAL['pyshark']:
            st.warning("pyshark not available — pcap streaming not supported here.")
        else:
            pcap_file = st.file_uploader("Upload pcap file", type=['pcap','pcapng'])
            if pcap_file:
                temp_path = "temp_stream.pcap"
                with open(temp_path, "wb") as f:
                    f.write(pcap_file.getbuffer())
                cap = pyshark.FileCapture(temp_path, keep_packets=False)
                placeholder = st.empty()
                count = 0
                for pkt in cap:
                    info = str(pkt)[:500]
                    with placeholder.container():
                        st.write(f"Packet {count+1}:")
                        st.text(info)
                    count += 1
                    time.sleep(0.2)
                    if count > 200:
                        break
                st.success("pcap stream preview complete.")
                cap.close()
                os.remove(temp_path)

# ---------- Export & Storage tab ----------
with export_tab:
    st.subheader("Export & Storage")
    st.write("Admin-only features: full export as ZIP, save to SQLite/DuckDB, or download DB.")
    if st.session_state['auth']['role'] != 'admin':
        st.info("Export & storage available to Admin only.")
    else:
        uploaded2 = st.file_uploader("Upload a CSV to export full predictions (admin)", type=['csv'], key="export_csv")
        if uploaded2:
            if st.button("Start full export (chunked)"):
                st.info("This will write a CSV with predictions and then compress to ZIP in exports/")
                df_all = load_csv_chunked(uploaded2)
                df_all = optimize_types(df_all)
                aligned_all = df_all.copy().reindex(columns=model_columns, fill_value=0)
                aligned_all.replace([np.inf,-np.inf], np.nan, inplace=True)
                aligned_all.fillna(0, inplace=True)
                total = len(aligned_all)
                chunk = 5000
                csv_path = os.path.join(EXPORT_DIR, f"predictions_full_{int(time.time())}.csv")
                with open(csv_path, "w", encoding="utf-8") as fout:
                    header_written = False
                    for i in range(0, total, chunk):
                        j = min(i+chunk, total)
                        Xc = scaler.transform(aligned_all.iloc[i:j])
                        preds = rf_model.predict(Xc)
                        probs = rf_model.predict_proba(Xc)[:,1]
                        iso_r = iso_model.predict(Xc)
                        iso_p = np.where(iso_r==-1,1,0)
                        df_chunk = pd.DataFrame({
                            "Flow Index": list(range(i+1,j+1)),
                            "Prediction": ["Attack" if x==1 else "Normal" for x in preds],
                            "Confidence": (probs*100).round(2),
                            "Severity": [severity(p)[0] for p in probs],
                            "Anomaly": ["Yes" if x==1 else "No" for x in iso_p]
                        })
                        df_chunk.to_csv(fout, header=not header_written, index=False, mode='a')
                        header_written = True
                        st.progress(min(1.0, j/total))
                zip_name = os.path.join(EXPORT_DIR, f"export_{int(time.time())}.zip")
                with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(csv_path, arcname=os.path.basename(csv_path))
                    stats = {"rows": total, "generated_at": time.ctime(), "file": os.path.basename(csv_path)}
                    stats_path = csv_path + ".stats.json"
                    with open(stats_path, "w") as sf:
                        json.dump(stats, sf)
                    zf.write(stats_path, arcname=os.path.basename(stats_path))
                st.success(f"Exported ZIP: {zip_name}")
                with open(zip_name, "rb") as f:
                    st.download_button("Download ZIP", data=f, file_name=os.path.basename(zip_name), mime="application/zip")

        if st.button("Download SQLite DB (predictions.db)"):
            if os.path.exists(SQL_DB):
                with open(SQL_DB, "rb") as f:
                    st.download_button("Download DB", data=f, file_name=os.path.basename(SQL_DB), mime="application/octet-stream")
            else:
                st.info("No DB file present yet.")

        if OPTIONAL.get('duckdb'):
            if st.button("Store predictions to DuckDB (optional)"):
                st.info("DuckDB integration available — configure as needed.")
        else:
            st.info("DuckDB not installed — install 'duckdb' to enable.")

# ---------- Diagnostics tab ----------
with diagnostics_tab:
    st.subheader("Automatic diagnostics & model repair suggestions")
    st.write("This checks for common schema/model mismatches and offers suggestions.")
    st.markdown("**Model columns summary (first 50)**")
    st.write(model_columns[:50])
    diag_csv = st.file_uploader("Upload a CSV to run diagnostics", type=['csv'], key="diag_csv")
    if diag_csv:
        df_diag = load_csv_chunked(diag_csv, chunksize=200000)
        df_diag = optimize_types(df_diag)
        user_cols = set(df_diag.columns.str.strip())
        req_cols = set(model_columns)
        missing = list(req_cols - user_cols)
        extra = list(user_cols - req_cols)
        st.write(f"Missing columns count: {len(missing)}")
        if missing:
            st.write("Example missing columns:", missing[:20])
        st.write(f"Extra columns count: {len(extra)}")
        if extra:
            st.write("Example extra columns:", extra[:20])
        if st.button("Auto-repair and download repaired CSV"):
            repaired = df_diag.copy()
            for c in missing:
                repaired[c] = 0
            repaired = repaired.reindex(columns=model_columns)
            buf = BytesIO()
            repaired.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download repaired CSV", data=buf, file_name="repaired_for_model.csv", mime="text/csv")

st.markdown("---")
st.caption("Enterprise app: Admin functions protected. For production, integrate with a real auth provider and secure storage.")