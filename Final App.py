# app_final_maps.py  -- Part 1/5
"""
Part 1/5: imports, configuration, caching helpers, geolocation hybrid
"""

import os
import time
import math
import json
import zipfile
import sqlite3
import secrets
from io import BytesIO
from typing import Tuple, List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Plotly for graphs
import plotly.express as px
import plotly.graph_objects as go

# Network graph
import networkx as nx

# Optional libraries
OPTIONAL = {}
try:
    import pydeck as pdk
    OPTIONAL['pydeck'] = True
except Exception:
    OPTIONAL['pydeck'] = False

try:
    import folium
    from streamlit_folium import st_folium
    OPTIONAL['folium'] = True
except Exception:
    OPTIONAL['folium'] = False

try:
    from ip2geotools.databases.noncommercial import DbIpCity
    OPTIONAL['ip2geo'] = True
except Exception:
    OPTIONAL['ip2geo'] = False

try:
    import geoip2.database
    OPTIONAL['geoip2'] = True
except Exception:
    OPTIONAL['geoip2'] = False

# pcap streaming
try:
    import pyshark
    OPTIONAL['pyshark'] = True
except Exception:
    OPTIONAL['pyshark'] = False

# scapy
try:
    import scapy.all as scapy
    OPTIONAL['scapy'] = True
except Exception:
    OPTIONAL['scapy'] = False

# file paths & model names (ensure your models are here)
USERS_FILE = None  # removed admin; placeholder
MODEL_FILES = {
    "rf": "rf_model.pkl",
    "iso": "iso_forest.pkl",
    "scaler": "scaler.pkl",
    "cols": "model_columns.pkl"
}
SQL_DB = "predictions.db"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Streamlit basic config
st.set_page_config(page_title="Cyber Threat Detection — Visuals & Maps", layout="wide", page_icon="🛰️")
st.title("Cyber Threat Detection — Visuals & Maps")
st.markdown("Prediction-only dashboard with rich visualizations and hybrid geolocation (GeoLite2 fallback → ip2geotools → deterministic).")

# Theme selector
st.sidebar.markdown("---")
THEME = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
PLOTLY_TEMPLATE = "plotly_dark" if THEME == "Dark" else "plotly_white"

# -------------------------
# Utility: robust query param helpers (lightweight)
# -------------------------
def get_qp():
    try:
        return st.query_params
    except Exception:
        if hasattr(st, "experimental_get_query_params"):
            return st.experimental_get_query_params()
        return {}

def set_qp(**kw):
    if hasattr(st, "experimental_set_query_params"):
        try:
            st.experimental_set_query_params(**kw)
            return
        except Exception:
            pass
    try:
        qp = st.query_params
        for k, v in kw.items():
            qp[k] = v
        st.session_state["_qp_update_ts"] = time.time()
    except Exception:
        pass

def clear_qp():
    if hasattr(st, "experimental_set_query_params"):
        try:
            st.experimental_set_query_params()
            return
        except Exception:
            pass
    try:
        st.query_params.clear()
        st.session_state["_qp_update_ts"] = time.time()
    except Exception:
        pass

# -------------------------
# Geolocation hybrid helper
# -------------------------
GEOIP2_PATH = "GeoLite2-City.mmdb"  # if you have GeoLite2, place it here

_geoip2_reader = None
if OPTIONAL.get('geoip2') and os.path.exists(GEOIP2_PATH):
    try:
        _geoip2_reader = geoip2.database.Reader(GEOIP2_PATH)
    except Exception:
        _geoip2_reader = None

def geoip2_lookup(ip: str):
    global _geoip2_reader
    if _geoip2_reader is None:
        return None
    try:
        rec = _geoip2_reader.city(ip)
        return rec.location.latitude, rec.location.longitude
    except Exception:
        return None

def ip2geotools_lookup(ip: str):
    if not OPTIONAL.get('ip2geo'):
        return None
    try:
        res = DbIpCity.get(ip, api_key="free")
        return float(res.latitude), float(res.longitude)
    except Exception:
        return None

def deterministic_ip_map(ip: str, seed_offset=0):
    # deterministic pseudo-random mapping from IP string -> lat/lon
    h = abs(hash(ip) + seed_offset) % (2**32)
    rng = np.random.RandomState(h)
    lat = float(rng.uniform(-55, 70))
    lon = float(rng.uniform(-180, 180))
    return lat, lon

def ip_to_latlon_hybrid(ip: str):
    """Try GeoLite2, then ip2geotools, else deterministic."""
    if not ip or str(ip).strip() == "":
        return None
    # try geoip2 first
    if _geoip2_reader is not None:
        out = geoip2_lookup(ip)
        if out:
            return out
    # try ip2geotools
    out = ip2geotools_lookup(ip)
    if out:
        return out
    # fallback deterministic
    return deterministic_ip_map(ip)

# -------------------------
# Caching: model loader and CSV loader
# -------------------------
@st.cache_resource
def load_models():
    """Load models from pickle files. Raise readable error if missing."""
    missing = [p for p in MODEL_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Place rf_model.pkl, iso_forest.pkl, scaler.pkl, model_columns.pkl in the app folder.")
    rf = pickle.load(open(MODEL_FILES['rf'], 'rb'))
    iso = pickle.load(open(MODEL_FILES['iso'], 'rb'))
    scaler = pickle.load(open(MODEL_FILES['scaler'], 'rb'))
    cols = pickle.load(open(MODEL_FILES['cols'], 'rb'))
    return rf, iso, scaler, cols

@st.cache_data(show_spinner=False)
def read_csv_chunked(uploaded_file, chunksize=200000):
    """Read CSV in chunks, return concatenated DataFrame (fast for large files)."""
    try:
        it = pd.read_csv(uploaded_file, chunksize=chunksize, low_memory=False)
        parts = []
        for ch in it:
            parts.append(ch)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        return df
    except Exception:
        return pd.read_csv(uploaded_file, low_memory=False)

def downcast_df(df: pd.DataFrame):
    """Downcast numeric types to reduce memory."""
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    return df

# -------------------------
# Load models (raise friendly error)
# -------------------------
try:
    rf_model, iso_model, scaler, model_columns = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -------------------------
# Severity scoring helper
# -------------------------
def severity_label(score: float) -> Tuple[str, str]:
    if score <= 0.20:
        return "Low", "#2ecc71"
    if score <= 0.50:
        return "Medium", "#f1c40f"
    if score <= 0.80:
        return "High", "#e67e22"
    return "Critical", "#e74c3c"
# app_final_maps.py  -- Part 2/5
"""
Part 2/5: Upload UI, preprocessing, alignment with model, lazy prediction, chunked export
"""

# -------------------------
# Alignment & prediction helpers
# -------------------------
def align_with_model(df: pd.DataFrame, model_cols: List[str]):
    """Return (aligned_df, original_df) where aligned has columns in model_cols."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    # drop obvious high-card columns that model didn't train on
    drops = ['Flow ID', 'FlowID', 'Timestamp', 'SimillarHTTP']
    df = df.drop(columns=[c for c in drops if c in df.columns], errors='ignore')
    aligned = df.reindex(columns=model_cols, fill_value=0)
    aligned.replace([np.inf, -np.inf], np.nan, inplace=True)
    aligned.fillna(0, inplace=True)
    return aligned, df

def predict_chunk(aligned_df: pd.DataFrame, start: int, end: int):
    """Predict rows [start:end) — returns predictions, probs, iso"""
    if start >= end:
        return np.array([]), np.array([]), np.array([])
    X = scaler.transform(aligned_df.iloc[start:end])
    preds = rf_model.predict(X)
    probs = rf_model.predict_proba(X)[:,1] if hasattr(rf_model, "predict_proba") else np.zeros(len(preds))
    iso_raw = iso_model.predict(X)
    iso_preds = np.where(iso_raw == -1, 1, 0)
    return preds, probs, iso_preds

# -------------------------
# Export full predictions function (chunked, memory-safe)
# -------------------------
def export_full_predictions(aligned_df: pd.DataFrame, out_csv_path: str, chunk_size=5000):
    total = len(aligned_df)
    header_written = False
    with open(out_csv_path, 'w', encoding='utf-8') as fout:
        for i in range(0, total, chunk_size):
            j = min(i+chunk_size, total)
            preds, probs, iso_preds = predict_chunk(aligned_df, i, j)
            df_chunk = pd.DataFrame({
                "Flow Index": list(range(i+1, j+1)),
                "Prediction": ["Attack" if p==1 else "Normal" for p in preds],
                "Confidence": (probs*100).round(2),
                "Severity": [severity_label(s)[0] for s in probs],
                "Anomaly": ["Yes" if x==1 else "No" for x in iso_preds]
            })
            df_chunk.to_csv(fout, header=not header_written, index=False, mode='a')
            header_written = True

# -------------------------
# UI: Upload & basic controls
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("Upload & Settings")
rows_per_page_default = st.sidebar.slider("Default rows per page", 10, 500, 50, step=10)
sample_size_default = st.sidebar.number_input("Default sampling size for estimates", min_value=0, max_value=100000, value=2000, step=100)

st.header("1) Upload CSV")
uploaded_file = st.file_uploader("Upload your network CSV (CICIDS-like). Use large CSVs — chunked reading is supported.", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to continue. Make sure your model files are present in the app folder.")
    st.stop()

with st.spinner("Reading CSV (chunked, optimized)..."):
    df_raw = read_csv_chunked(uploaded_file)

if df_raw is None or df_raw.empty:
    st.error("Uploaded CSV appears empty or unreadable.")
    st.stop()

st.success(f"Loaded {len(df_raw):,} rows.")
st.dataframe(df_raw.head())

# downcast numeric types
df_raw = downcast_df(df_raw)

# align to model
aligned_df, original_df = align_with_model(df_raw, model_columns)
n_rows = len(aligned_df)
st.info(f"Data aligned to model columns: {n_rows:,} rows. Missing features (if any) were zero-filled.")

# page controls
st.subheader("View & Predict (lazy)")
rows_per_page = st.slider("Rows per page (viewer)", 5, 500, rows_per_page_default, step=5)
total_pages = max(1, math.ceil(n_rows / rows_per_page))
page = st.number_input("Page number", min_value=1, max_value=total_pages, value=1)
start_idx = (page - 1) * rows_per_page
end_idx = min(start_idx + rows_per_page, n_rows)

# sample-based estimate
sample_size = st.number_input("Sampling size for estimate (0 = off)", min_value=0, max_value=100000, value=sample_size_default, step=100)
est_attacks = None
if sample_size > 0:
    s = min(sample_size, n_rows)
    idxs = np.random.RandomState(42).choice(n_rows, s, replace=False)
    Xs = scaler.transform(aligned_df.iloc[idxs])
    sample_preds = rf_model.predict(Xs)
    est_attacks = int(sample_preds.mean() * n_rows)
    st.info(f"Estimated total attacks (sample size {s}): {est_attacks} (ratio {sample_preds.mean():.3f})")

# predict only current page
with st.spinner("Predicting current page..."):
    preds_page, probs_page, iso_page = predict_chunk(aligned_df, start_idx, end_idx)

# assemble page df
page_df = pd.DataFrame({
    "Flow Index": list(range(start_idx+1, end_idx+1)),
    "Prediction": ["Attack" if p==1 else "Normal" for p in preds_page],
    "Confidence %": (probs_page * 100).round(2),
    "Severity": [severity_label(p)[0] for p in probs_page],
    "Anomaly": ["Yes" if x==1 else "No" for x in iso_page]
})

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows on page", f"{start_idx+1} - {end_idx}")
k2.metric("Attacks on page", int(((page_df['Prediction']=="Attack").sum())))
k3.metric("Estimated attacks (sample)", est_attacks if est_attacks is not None else "N/A")
k4.metric("Avg confidence (page)", f"{np.mean(probs_page):.3f}" if len(probs_page)>0 else "0.000")

# display table with correct counts (fix for 100% issue)
def colorize_row(r):
    # color by severity
    sev = r["Severity"]
    colors = {"Low":"#d5f5e3","Medium":"#fff9dd","High":"#ffe6c6","Critical":"#ffd6d6"}
    return [f"background-color: {colors.get(sev, '#ffffff')}"]*len(r)

st.markdown("### Predictions (current page)")
st.dataframe(page_df.style.apply(colorize_row, axis=1), use_container_width=True)

# Save preview
if st.button("Save current page predictions to temporary CSV"):
    temp_path = os.path.join(EXPORT_DIR, f"page_predictions_{int(time.time())}.csv")
    page_df.to_csv(temp_path, index=False)
    st.success(f"Saved to {temp_path}")
    with open(temp_path, "rb") as f:
        st.download_button("Download page CSV", data=f, file_name=os.path.basename(temp_path), mime="text/csv")

# Export full predictions (chunked)
st.subheader("Export full predictions (chunked)")
if st.button("Export full predictions to ZIP"):
    out_csv = os.path.join(EXPORT_DIR, f"predictions_full_{int(time.time())}.csv")
    with st.spinner("Generating full predictions (chunked)..."):
        export_full_predictions(aligned_df, out_csv, chunk_size=5000)
    zip_path = os.path.join(EXPORT_DIR, f"predictions_export_{int(time.time())}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=os.path.basename(out_csv))
        stats = {"rows": n_rows, "generated_at": time.ctime(), "file": os.path.basename(out_csv)}
        stats_path = out_csv + ".stats.json"
        with open(stats_path, "w") as sf:
            json.dump(stats, sf)
        zf.write(stats_path, arcname=os.path.basename(stats_path))
    st.success(f"Export created: {zip_path}")
    with open(zip_path, "rb") as f:
        st.download_button("Download export ZIP", data=f, file_name=os.path.basename(zip_path), mime="application/zip")
# app_final_maps.py  -- Part 3/5
"""
Part 3/5: Graphical Dashboard — Pie, Bar, Heatmap, Correlation matrix, 3D scatter, Node topology
"""

st.markdown("---")
st.header("2) Visual Dashboard")

# Accurate Attack vs Normal pie (use counts from page or global sample)
use_global_for_visuals = st.checkbox("Use full dataset sampling for visuals (slower)", value=False)

if use_global_for_visuals:
    sample_n_vis = st.number_input("Sample size for visuals", min_value=100, max_value=min(50000, n_rows), value=5000, step=100)
    idxs = np.random.RandomState(1).choice(n_rows, sample_n_vis, replace=False)
    X_vis = scaler.transform(aligned_df.iloc[idxs])
    preds_vis = rf_model.predict(X_vis)
    probs_vis = rf_model.predict_proba(X_vis)[:,1]
    df_vis = pd.DataFrame({"pred": ["Attack" if p==1 else "Normal" for p in preds_vis], "score": probs_vis})
else:
    df_vis = pd.DataFrame({"pred": page_df["Prediction"], "score": page_df["Confidence %"]/100.0})

# Pie chart (Attack vs Normal) with proper counts
counts = df_vis['pred'].value_counts().reindex(["Attack","Normal"], fill_value=0).reset_index()
counts.columns = ['Type','Count']
fig_pie = px.pie(counts, names='Type', values='Count', color='Type',
                 color_discrete_map={'Attack':'red','Normal':'green'}, template=PLOTLY_TEMPLATE,
                 title="Attack vs Normal")
st.plotly_chart(fig_pie, use_container_width=True)

# Bar chart: Severity distribution (accurate)
sev_counts = None
if use_global_for_visuals:
    sev_labels = [severity_label(s)[0] for s in df_vis['score']]
    sev_counts = pd.Series(sev_labels).value_counts().reindex(["Critical","High","Medium","Low"], fill_value=0).reset_index()
    sev_counts.columns = ['Severity','Count']
else:
    sev_counts = page_df['Severity'].value_counts().reindex(["Critical","High","Medium","Low"], fill_value=0).reset_index()
    sev_counts.columns = ['Severity','Count']
fig_bar = px.bar(sev_counts, x='Severity', y='Count', color='Severity', template=PLOTLY_TEMPLATE, title="Severity distribution")
st.plotly_chart(fig_bar, use_container_width=True)

# Correlation heatmap (top numeric features sampled)
st.markdown("### Correlation heatmap (sampled numeric features)")
numeric_cols = [c for c in aligned_df.columns if np.issubdtype(aligned_df[c].dtype, np.number)]
if len(numeric_cols) >= 4:
    sample_corr_n = min(4000, n_rows)
    idxs_corr = np.random.RandomState(2).choice(n_rows, sample_corr_n, replace=False)
    corr_df = aligned_df.iloc[idxs_corr][numeric_cols].sample(min(len(numeric_cols), 25), axis=1)
    corr = corr_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Feature correlation (sampled)", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric features for correlation heatmap.")

# 3D scatter (Duration vs PacketCount vs Score) — choose good numeric cols automatically
st.markdown("### 3D Threat scatter (sampled)")
# choose 2 numeric columns for X/Y that appear significant: try "Flow Duration" "Total Fwd Packets" etc.
preferred = ["Flow Duration", "Flow.Duration", "Total Fwd Packets", "Total Backward Packets", "Fwd Packet Length Max", "Bwd Packet Length Max"]
found = [c for c in preferred if c in aligned_df.columns]
if len(found) >= 2:
    use_cols = found[:2]
else:
    # fallback to first 2 numeric columns
    use_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else None

if use_cols is not None:
    sample_n_3d = min(5000, n_rows)
    sidx = np.random.RandomState(3).choice(n_rows, sample_n_3d, replace=False)
    sample_3d = aligned_df.iloc[sidx]
    Xs = scaler.transform(sample_3d)
    probs_3d = rf_model.predict_proba(Xs)[:,1] if hasattr(rf_model, "predict_proba") else np.zeros(len(Xs))
    fig3d = px.scatter_3d(sample_3d, x=use_cols[0], y=use_cols[1], z=probs_3d,
                         color=["Attack" if p>0.5 else "Normal" for p in probs_3d],
                         color_discrete_map={'Attack':'red','Normal':'green'}, template=PLOTLY_TEMPLATE, title="3D threat scatter (sampled)")
    fig3d.update_layout(height=700)
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.info("No suitable numeric columns for 3D scatter.")

# Node topology: build source->dest edges if columns exist
st.markdown("### Network topology (node graph)")
# heuristics to find source and destination columns
src_cols = [c for c in original_df.columns if 'src' in c.lower() or 'source' in c.lower()]
dst_cols = [c for c in original_df.columns if 'dst' in c.lower() or 'dest' in c.lower() or 'destination' in c.lower()]

if src_cols and dst_cols:
    src_col = src_cols[0]
    dst_col = dst_cols[0]
    st.info(f"Using source column `{src_col}` and destination column `{dst_col}` for topology.")
    sample_n_nodes = min(3000, n_rows)
    sidx = np.random.RandomState(4).choice(n_rows, sample_n_nodes, replace=False)
    edges = original_df.iloc[sidx][[src_col, dst_col]].dropna()
    # build edge counts
    edge_counts = edges.groupby([src_col, dst_col]).size().reset_index(name='count').sort_values('count', ascending=False).head(500)
    G = nx.DiGraph()
    for _, row in edge_counts.iterrows():
        s = str(row[src_col]); d = str(row[dst_col]); w = int(row['count'])
        G.add_node(s); G.add_node(d)
        G.add_edge(s, d, weight=w)
    # layout with spring layout (may be slow on many nodes)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=5)
    # build plotly scatter for nodes and edges
    edge_x = []
    edge_y = []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = []
    node_y = []
    node_text = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(n))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center',
                            marker=dict(size=8, color='cyan'), hoverinfo='text')
    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Network topology (sampled)", showlegend=False, template=PLOTLY_TEMPLATE))
    fig_net.update_layout(height=700)
    st.plotly_chart(fig_net, use_container_width=True)
else:
    st.info("No clear source/destination columns found for topology. Columns checked: src-like and dst-like heuristics.")
# app_final_maps.py  -- Part 4/5
"""
Part 4/5: Maps — IP geolocation hybrid map, node-to-node map, and world heatmap
"""

st.markdown("---")
st.header("3) Maps & Geo Visualizations")

# choose IP-like column
ip_candidates = [c for c in original_df.columns if 'ip' in c.lower() or 'addr' in c.lower() or 'host' in c.lower()]
ip_col = None
if ip_candidates:
    ip_col = ip_candidates[0]
else:
    # heuristic: any column with dotted values
    for c in original_df.columns:
        try:
            vals = original_df[c].astype(str).dropna().head(20).tolist()
            if vals and all('.' in v for v in vals if v.strip()):
                ip_col = c
                break
        except Exception:
            continue

st.subheader("A) IP → GeoLocation World Map")
st.write("Hybrid geolocation: GeoLite2 (if available) → ip2geotools → deterministic fallback.")

if ip_col is None:
    st.info("No IP-like column detected — you can upload an IP->LatLon mapping file or the map will use deterministic pseudo-locations.")
else:
    st.info(f"Using `{ip_col}` column for geolocation.")
    max_points = st.slider("Max points to geolocate (for speed)", 10, 2000, 300, step=10)
    limit = min(len(original_df), max_points)
    sample_idx = np.random.RandomState(42).choice(len(original_df), limit, replace=False)
    ip_list = original_df[ip_col].astype(str).iloc[sample_idx].fillna("").tolist()
    coords = []
    with st.spinner("Resolving IPs (hybrid)..."):
        for ip in ip_list:
            try:
                latlon = ip_to_latlon_hybrid(ip)
            except Exception:
                latlon = None
            if latlon:
                coords.append({"ip": ip, "lat": latlon[0], "lon": latlon[1]})
    map_df = pd.DataFrame(coords)
    if map_df.empty:
        st.info("No coordinates resolved.")
    else:
        st.write(f"Resolved {len(map_df)} points.")
        # pydeck if available
        if OPTIONAL.get('pydeck'):
            layer = pdk.Layer("ScatterplotLayer", map_df, get_position='[lon, lat]', get_radius=20000, get_fill_color='[255,0,0,160]', pickable=True)
            view = pdk.ViewState(latitude=float(map_df['lat'].mean()), longitude=float(map_df['lon'].mean()), zoom=1.5)
            r = pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/dark-v9')
            st.pydeck_chart(r)
        elif OPTIONAL.get('folium'):
            m = folium.Map(location=[map_df['lat'].mean(), map_df['lon'].mean()], tiles='CartoDB dark_matter', zoom_start=1)
            for _, r in map_df.iterrows():
                folium.CircleMarker(location=[r['lat'], r['lon']], radius=5, color='red', fill=True).add_to(m)
            st_folium(m, width=800, height=400)
        else:
            # basic st.map
            md = map_df.rename(columns={'lat':'latitude', 'lon':'longitude'})
            st.map(md[['latitude','longitude']])

# Node-to-node map overlay (plot links between resolved IPs)
st.subheader("B) Node-to-Node Geo Map (if source/dest columns exist)")
if ip_col is None:
    st.info("No IP column: node-to-node geo map will be synthesized from topology using deterministic mapping.")
# find src/dst again
src_cols = [c for c in original_df.columns if 'src' in c.lower() or 'source' in c.lower()]
dst_cols = [c for c in original_df.columns if 'dst' in c.lower() or 'dest' in c.lower() or 'destination' in c.lower()]
if src_cols and dst_cols:
    s_col = src_cols[0]; d_col = dst_cols[0]
    max_links = st.slider("Max links to plot", 10, 1000, 200, step=10)
    sample_n = min(2000, len(original_df))
    sidx = np.random.RandomState(8).choice(len(original_df), sample_n, replace=False)
    edges_df = original_df.iloc[sidx][[s_col, d_col]].dropna()
    edge_counts = edges_df.groupby([s_col, d_col]).size().reset_index(name='count').sort_values('count', ascending=False).head(max_links)
    # build coordinates for each unique node (try to geolocate)
    nodes = pd.unique(edge_counts[[s_col,d_col]].values.ravel('K'))
    node_coords = {}
    for n in nodes:
        if ip_col and n in original_df[ip_col].astype(str).values:
            latlon = ip_to_latlon_hybrid(n)
        else:
            latlon = deterministic_ip_map(str(n))
        node_coords[n] = latlon
    # assemble links with lat/lon
    links = []
    for _, row in edge_counts.iterrows():
        s_n, d_n = row[s_col], row[d_col]
        s_latlon = node_coords.get(s_n)
        d_latlon = node_coords.get(d_n)
        if s_latlon and d_latlon:
            links.append({"s": s_n, "d": d_n, "s_lat": s_latlon[0], "s_lon": s_latlon[1], "d_lat": d_latlon[0], "d_lon": d_latlon[1], "count": int(row['count'])})
    links_df = pd.DataFrame(links)
    if links_df.empty:
        st.info("No links to display on map.")
    else:
        if OPTIONAL.get('pydeck'):
            # build arcs
            layer = pdk.Layer(
                "ArcLayer",
                data=links_df,
                get_source_position=["s_lon", "s_lat"],
                get_target_position=["d_lon", "d_lat"],
                get_width="count",
                get_source_color=[255, 140, 0],
                get_target_color=[0, 128, 255],
                pickable=True
            )
            view = pdk.ViewState(latitude=float(links_df[['s_lat','d_lat']].stack().mean()), longitude=float(links_df[['s_lon','d_lon']].stack().mean()), zoom=1.0)
            deck = pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/dark-v9')
            st.pydeck_chart(deck)
        elif OPTIONAL.get('folium'):
            m = folium.Map(location=[links_df[['s_lat','d_lat']].stack().mean(), links_df[['s_lon','d_lon']].stack().mean()], tiles='CartoDB dark_matter', zoom_start=1)
            for _, r in links_df.iterrows():
                folium.PolyLine(locations=[(r['s_lat'], r['s_lon']), (r['d_lat'], r['d_lon'])], color='orange', weight=max(1, min(10, r['count']))).add_to(m)
            st_folium(m, width=900, height=500)
        else:
            st.map(pd.DataFrame({"latitude": links_df['s_lat'].tolist() + links_df['d_lat'].tolist(), "longitude": links_df['s_lon'].tolist() + links_df['d_lon'].tolist()}))

# World heatmap: aggregate counts per country (if geolocation possible) else use deterministic bucket by lat/lon
st.subheader("C) World Traffic Heatmap (density)")
heat_sample = st.slider("Heatmap sample size", 100, min(20000, n_rows), 5000, step=100)
sidx = np.random.RandomState(99).choice(n_rows, min(heat_sample, n_rows), replace=False)
heat_ips = []
if ip_col:
    sampled_ips = original_df[ip_col].astype(str).iloc[sidx].fillna("").tolist()
    for ip in sampled_ips:
        latlon = ip_to_latlon_hybrid(ip)
        if latlon:
            heat_ips.append(latlon)
else:
    # deterministic seed mapping of a column (e.g., Flow index)
    for i in sidx:
        heat_ips.append(deterministic_ip_map(str(i)))
if not heat_ips:
    st.info("Insufficient geolocation data for heatmap.")
else:
    heat_df = pd.DataFrame(heat_ips, columns=['lat','lon'])
    if OPTIONAL.get('pydeck'):
        layer = pdk.Layer(
            "HeatmapLayer",
            heat_df,
            get_position='[lon, lat]',
            aggregation='MEAN',
            radiusPixels=60
        )
        view = pdk.ViewState(latitude=float(heat_df['lat'].mean()), longitude=float(heat_df['lon'].mean()), zoom=1.0)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/dark-v9'))
    elif OPTIONAL.get('folium'):
        m = folium.Map(location=[heat_df['lat'].mean(), heat_df['lon'].mean()], tiles='CartoDB positron', zoom_start=1)
        from folium.plugins import HeatMap
        HeatMap(heat_df.values.tolist(), radius=10).add_to(m)
        st_folium(m, width=900, height=500)
    else:
        st.map(heat_df.rename(columns={'lat':'latitude','lon':'longitude'})[['latitude','longitude']])
# app_final_maps.py  -- Part 5/5
"""
Part 5/5: Diagnostics, finishing touches, and instructions
"""

st.markdown("---")
st.header("4) Diagnostics & Model Info")

# Show model info and columns
st.subheader("Model summary")
try:
    # Show a brief model summary
    rf = rf_model
    st.write("RandomForest model info:")
    st.write(f"- Estimators: {getattr(rf, 'n_estimators', 'N/A')}")
    st.write(f"- Features expected: {len(model_columns)}")
    st.write("First 50 model columns:")
    st.write(model_columns[:50])
except Exception:
    st.info("Could not render model summary.")

# Diagnostics: upload file to run repair suggestion
st.subheader("Diagnostics: Auto-repair missing columns")
diag_upload = st.file_uploader("Upload CSV to diagnose / repair (optional)", type=["csv"], key="diag_upload")
if diag_upload:
    df_diag = read_csv_chunked(diag_upload)
    df_diag = downcast_df(df_diag)
    user_cols = set(df_diag.columns.str.strip())
    model_cols_set = set(model_columns)
    missing_cols = list(model_cols_set - user_cols)
    extra_cols = list(user_cols - model_cols_set)
    st.write(f"Missing columns: {len(missing_cols)}")
    st.write(f"Extra columns: {len(extra_cols)}")
    if missing_cols:
        st.write("Example missing:", missing_cols[:10])
        if st.button("Auto-repair & download repaired CSV"):
            repaired = df_diag.copy()
            for c in missing_cols:
                repaired[c] = 0
            repaired = repaired.reindex(columns=model_columns)
            buf = BytesIO()
            repaired.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download repaired CSV", data=buf, file_name="repaired_for_model.csv", mime="text/csv")
    else:
        st.success("No missing columns detected.")
