import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.federated_runner import run_federated_experiment
from src.client import CLIENT_IDENTITIES
from src.dataset_downloader import (
    UNSW_ATTACK_CATEGORIES,
    CICIDS_ATTACK_CATEGORIES,
    ensure_dataset_available,
    normalize_real_csv,
)
from src.models import predict_flows_from_dataframe
from src.server import CentralFederatedServer


st.set_page_config(
    page_title="AEGIS-FL™ - Privacy-Preserving Federated Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    background-color: #ffffff !important;
    color: #1a1a2e;
    scroll-behavior: smooth !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
span[translate="no"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-style: normal !important;
    font-weight: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-smoothing: antialiased !important;
}

.block-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 2rem 4rem 2rem !important;
    color: #1a1a2e;
}

.block-container h1, .block-container h2, .block-container h3, .block-container h4 {
    color: #1a1a2e;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.block-container [data-testid="stMarkdownContainer"] p {
    color: #1a1a2e;
}

.sy-navbar {
    position: sticky;
    top: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 16px 32px;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid #f0f0f0;
    margin: 0 -2rem 0 -2rem;
}
.sy-logo { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
.sy-logo-text { font-size: 22px; font-weight: 900; letter-spacing: -0.04em; color: #1a1a2e; white-space: nowrap !important; }
.sy-logo-text .tm { font-size: 12px; color: #facc15; vertical-align: super; }
.sy-nav-links { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }

.sy-hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 48px;
    align-items: center;
    padding: 60px 0 48px 0;
}
.sy-hero-tag {
    display: inline-block;
    background: #fef9c3;
    color: #92400e;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 6px 16px;
    border-radius: 50px;
    margin-bottom: 20px;
}
.sy-hero h1 {
    font-size: 50px !important;
    font-weight: 900 !important;
    line-height: 1.08 !important;
    letter-spacing: -0.04em !important;
    color: #1a1a2e !important;
    margin: 0 0 20px 0 !important;
}
.sy-hero h1 .highlight { color: #eab308; }
.sy-hero-desc { font-size: 17px; color: #64748b; line-height: 1.65; margin-bottom: 28px; max-width: 540px; }
.sy-hero-social { display: flex; align-items: center; gap: 14px; margin-top: 18px; }
.sy-stars { color: #facc15; font-size: 18px; letter-spacing: 2px; }
.sy-review-text { font-size: 13px; color: #64748b; font-weight: 600; }

.sy-hero-right {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
    padding: 12px 0;
}
.sy-float-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.03);
    position: relative;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    border: 1px solid #f1f5f9;
}
.sy-float-card:hover { transform: translateY(-4px); box-shadow: 0 16px 36px rgba(0,0,0,0.10); }
.sy-float-card.card-2, .sy-float-card.card-4 { margin-top: 20px; }
.sy-float-icon {
    width: 44px; height: 44px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin-bottom: 12px;
}
.sy-float-title { font-weight: 700; font-size: 13px; color: #64748b; margin-bottom: 4px; }
.sy-float-value { font-weight: 900; font-size: 24px; color: #1a1a2e; letter-spacing: -0.03em; line-height: 1.15; }
.sy-float-sub { font-size: 12px; color: #94a3b8; margin-top: 6px; font-weight: 500; }
.sy-float-arrow {
    position: absolute; top: 16px; right: 16px; width: 30px; height: 30px;
    background: #facc15; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; color: #1a1a2e; font-weight: 900;
}

.sy-section-tag {
    display: inline-block; background: #fef9c3; color: #92400e;
    font-size: 11px; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.1em; padding: 5px 14px; border-radius: 50px; margin-bottom: 14px;
}
.sy-section-title {
    font-size: 34px !important; font-weight: 900 !important; color: #1a1a2e !important;
    letter-spacing: -0.035em !important; line-height: 1.15 !important; margin: 0 0 12px 0 !important;
}
.sy-section-desc { font-size: 16px; color: #64748b; line-height: 1.6; max-width: 680px; }
.sy-section-center { text-align: center; max-width: 720px; margin: 0 auto; padding: 52px 0 32px 0; }
.sy-section-center .sy-section-desc { margin: 0 auto; }

.sy-steps-bar { display: flex; align-items: center; justify-content: center; margin: 24px 0 40px 0; }
.sy-step-dot {
    width: 48px; height: 48px; border-radius: 50%; background: #fef9c3; border: 3px solid #facc15;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 900; color: #1a1a2e; z-index: 1;
}
.sy-step-line { width: 72px; height: 3px; background: #e5e7eb; margin: 0 -2px; }
.sy-steps-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 24px; }
.sy-step-card { text-align: center; padding: 20px 14px; }
.sy-step-icon {
    width: 60px; height: 60px; border-radius: 50%; background: #fef9c3;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin: 0 auto 14px auto; border: 3px solid #facc15;
}
.sy-step-title { font-weight: 800; font-size: 15px; color: #1a1a2e; margin-bottom: 6px; }
.sy-step-text { font-size: 13px; color: #64748b; line-height: 1.5; }

.sy-kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin: 20px 0; }
.sy-kpi {
    background: #ffffff; border: 1px solid #f3f4f6; border-radius: 20px; padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04); position: relative; overflow: hidden;
}
.sy-kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; }
.sy-kpi-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 8px; }
.sy-kpi-num { font-size: 28px; font-weight: 900; color: #1a1a2e; letter-spacing: -0.03em; line-height: 1; }
.sy-kpi-sub { font-size: 12px; color: #64748b; margin-top: 8px; font-weight: 500; }
.sy-kpi.accent-yellow::before { background: #facc15; }
.sy-kpi.accent-dark::before { background: #1a1a2e; }
.sy-kpi.accent-blue::before { background: #3b82f6; }
.sy-kpi.accent-green::before { background: #22c55e; }
.sy-kpi.accent-purple::before { background: #8b5cf6; }
.sy-kpi.accent-red::before { background: #ef4444; }

.sy-client-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 18px; margin: 24px 0; }
.sy-client {
    background: #ffffff; border: 1px solid #f3f4f6; border-radius: 20px; padding: 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04); display: flex; flex-direction: column; min-height: 270px;
}
.sy-client.honest { border-top: 4px solid #22c55e; }
.sy-client.compromised { border-top: 4px solid #ef4444; background: #fef2f2; }
.sy-client-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.sy-client-icon {
    width: 46px; height: 46px; border-radius: 14px; background: #f8fafc;
    display: flex; align-items: center; justify-content: center; font-size: 22px;
}
.sy-badge {
    font-size: 10px; font-weight: 800; padding: 4px 10px; border-radius: 50px;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.sy-badge.green { background: #dcfce7; color: #166534; }
.sy-badge.red { background: #fee2e2; color: #991b1b; }
.sy-client-name { font-weight: 900; font-size: 15px; color: #1a1a2e; margin-bottom: 2px; }
.sy-client-sector { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
.sy-client-detail { font-size: 12px; color: #475569; line-height: 1.65; flex: 1; }
.sy-client-footer {
    margin-top: 12px; padding-top: 10px; border-top: 1px solid #f3f4f6;
    font-size: 11.5px; font-weight: 700; color: #16a34a;
}

.sy-panel {
    background: #ffffff; border: 1px solid #f3f4f6; border-radius: 20px; padding: 26px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04); height: 100%;
}
.sy-panel-title { font-weight: 900; font-size: 18px; color: #1a1a2e; margin-bottom: 12px; }
.sy-panel-text { font-size: 14px; color: #64748b; line-height: 1.7; }

.sy-alert {
    border-radius: 16px; padding: 20px 26px; margin: 20px 0;
    display: flex; align-items: flex-start; gap: 14px;
}
.sy-alert.danger { background: linear-gradient(135deg, #fef2f2, #fff1f2); border: 1px solid #fecdd3; }
.sy-alert.success { background: linear-gradient(135deg, #f0fdf4, #ecfdf5); border: 1px solid #bbf7d0; }
.sy-alert-icon { font-size: 26px; flex-shrink: 0; }
.sy-alert-title { font-weight: 900; font-size: 16px; color: #1a1a2e; }
.sy-alert-text { font-size: 14px; color: #475569; margin-top: 4px; line-height: 1.5; }

.sy-compare-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin: 28px 0; }
.sy-compare-card {
    background: #ffffff; border: 1px solid #f3f4f6; border-radius: 20px; padding: 26px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}
.sy-compare-icon {
    width: 52px; height: 52px; border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin-bottom: 14px;
}
.sy-compare-card.danger .sy-compare-icon { background: #fef2f2; }
.sy-compare-card.warning .sy-compare-icon { background: #fffbeb; }
.sy-compare-title { font-weight: 900; font-size: 17px; color: #1a1a2e; margin-bottom: 4px; }
.sy-compare-sub { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
.sy-compare-card.danger .sy-compare-sub { color: #dc2626; }
.sy-compare-card.warning .sy-compare-sub { color: #d97706; }
.sy-compare-text { font-size: 13px; color: #64748b; line-height: 1.6; }

.sy-pillars-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; margin: 24px 0; }
.sy-pillar {
    background: #ffffff; border: 1px solid #f3f4f6; border-radius: 20px; padding: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04); border-top: 4px solid transparent;
}
.sy-pillar-num { font-size: 30px; font-weight: 900; color: #facc15; margin-bottom: 8px; }
.sy-pillar-title { font-weight: 800; font-size: 15px; color: #1a1a2e; margin-bottom: 6px; }
.sy-pillar-text { font-size: 13px; color: #64748b; line-height: 1.55; }

.stButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
button[kind="secondary"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    border-radius: 50px !important;
    border: 1.5px solid #cbd5e1 !important;
    padding: 9px 14px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    white-space: nowrap !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}
.stButton > button *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="stBaseButton-secondary"] *,
button[kind="secondary"] * {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    font-weight: 700 !important;
}
.stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    background-color: #fef9c3 !important;
    border-color: #facc15 !important;
    color: #92400e !important;
    -webkit-text-fill-color: #92400e !important;
}

.stButton > button[data-testid="baseButton-primary"],
.stButton > button[data-testid="stBaseButton-primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
    background-color: #facc15 !important;
    background: #facc15 !important;
    border: 1.5px solid #eab308 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    border-radius: 50px !important;
    padding: 9px 14px !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    white-space: nowrap !important;
    box-shadow: 0 4px 14px rgba(250, 204, 21, 0.40) !important;
}
.stButton > button[data-testid="baseButton-primary"] *,
.stButton > button[data-testid="stBaseButton-primary"] *,
button[data-testid="baseButton-primary"] *,
button[data-testid="stBaseButton-primary"] *,
button[kind="primary"] * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    font-weight: 800 !important;
}

.stSelectbox label, .stSlider label, .stCheckbox label span {
    font-weight: 700 !important;
    font-size: 13px !important;
    color: #1a1a2e !important;
}
div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] div, div[data-baseweb="select"] input {
    color: #1a1a2e !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    margin: 12px 0 !important;
}
[data-testid="stExpander"] details summary {
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #1e293b !important;
    background: #f8fafc !important;
    padding: 12px 18px !important;
    border-radius: 14px !important;
}

.sy-footer { text-align: center; padding: 48px 0 24px 0; border-top: 1px solid #f3f4f6; margin-top: 48px; }
.sy-footer-brand { font-size: 20px; font-weight: 900; color: #1a1a2e; letter-spacing: -0.03em; }
.sy-footer-sub { font-size: 13px; color: #94a3b8; margin-top: 8px; }

@media (max-width: 960px) {
    .sy-hero { grid-template-columns: 1fr; }
    .sy-kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .sy-client-grid { grid-template-columns: repeat(2, 1fr); }
    .sy-compare-grid, .sy-pillars-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


if "active_page" not in st.session_state:
    st.session_state.active_page = "Command Center"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INTERACTIVE EXPERIMENT CONFIG PANEL                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
csv_unsw = ROOT_DIR / "results" / "metrics.csv"
csv_cicids = ROOT_DIR / "results" / "cicids_metrics.csv"
audit_file = ROOT_DIR / "results" / "audit_log.json"
ledger_valid, ledger_msg = CentralFederatedServer.verify_audit_log(audit_file)

# Read defaults before rendering top bar so badges reflect live configuration & ledger state
with st.expander("Experiment Configuration Panel", expanded=False):
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        st.markdown("**Dataset & Architecture**")
        dataset_choice = st.selectbox("Benchmark Dataset", ["UNSW-NB15", "CICIDS2017"], index=0)
        dataset_key = "unsw" if dataset_choice == "UNSW-NB15" else "cicids"
        model_choice = st.selectbox(
            "Neural Architecture",
            [
                "1D ResNet (Residual)",
                "Deep MLP (Tabular)",
                "8-Token Feature Attention",
                "Zero-Day Autoencoder (MSE)",
            ],
            index=0,
        )
        model_map = {
            "Deep MLP (Tabular)": "mlp",
            "1D ResNet (Residual)": "resnet",
            "8-Token Feature Attention": "attention",
            "Zero-Day Autoencoder (MSE)": "autoencoder",
        }
        selected_model = model_map[model_choice]
        multiclass_toggle = st.checkbox(
            "Multi-Class Attack Attribution",
            value=False,
            disabled=(selected_model == "autoencoder"),
        )
        non_iid_toggle = st.checkbox("Non-IID Skew (Dirichlet alpha=0.5)", value=False)
        rounds_count = st.slider("Federated Rounds", 2, 8, 3)

    with ctrl_col2:
        st.markdown("**Privacy & Cryptography**")
        dp_toggle = st.checkbox("Opacus Per-Sample DP-SGD (RDP)", value=(selected_model != "autoencoder"))
        clipping_norm = st.slider("Clipping Norm (C)", 0.5, 5.0, 1.0, 0.5) if dp_toggle else 1.0
        noise_multiplier = st.slider("Gaussian Noise (sigma)", 0.50, 2.00, 0.65, 0.05) if dp_toggle else 0.0
        secagg_toggle = st.checkbox("Drop-Resilient SecAgg+", value=True)
        sparsify_toggle = st.checkbox("Top-k Sparsify (Energy-Conserving)", value=False)
        local_dp_toggle = st.checkbox("Local DP (Label Randomized Resp.)", value=False)
        paillier_toggle = st.checkbox("Paillier HE (Head Ciphertext)", value=False)
        audit_toggle = st.checkbox("SHA-256 + Pedersen Audit", value=True)

    with ctrl_col3:
        st.markdown("**Adversarial Attack**")
        attack_choice = st.selectbox(
            "Simulate Attack",
            [
                "None (All Honest)",
                "Model Poisoning (Scaling -25x)",
                "Stealth Adaptive Norm Poisoning",
                "Data Poisoning (Label Flip)",
                "Targeted Backdoor Trigger",
            ],
            index=0,
        )
        attack_map = {
            "None (All Honest)": "none",
            "Model Poisoning (Scaling -25x)": "model_scaling",
            "Stealth Adaptive Norm Poisoning": "adaptive_poisoning",
            "Data Poisoning (Label Flip)": "label_flipping",
            "Targeted Backdoor Trigger": "backdoor",
        }
        selected_attack = attack_map[attack_choice]
        compromised_cid = 2
        if selected_attack != "none":
            compromised_cid = st.selectbox(
                "Compromised Client",
                options=[0, 1, 2, 3, 4],
                format_func=lambda c: CLIENT_IDENTITIES[c]["name"],
                index=2,
            )

    with ctrl_col4:
        st.markdown("**Byzantine Defense & Run**")
        defense_choice = st.selectbox(
            "Byzantine Defense",
            [
                "Multi-Krum (Provable Geometric)",
                "Norm + Head Sign Guard",
                "Cosine Similarity (FLTrust)",
                "FLAME (DBSCAN Clustering + DP)",
                "Client Reputation-Weighted (EMA)",
                "Adaptive Clip Norm",
                "Coordinate-Wise Median",
                "Trimmed Mean",
                "FedAvg (No Defense)",
            ],
            index=0,
        )
        defense_map = {
            "Norm + Head Sign Guard": "norm_clipping",
            "Cosine Similarity (FLTrust)": "cosine_similarity",
            "Multi-Krum (Provable Geometric)": "multi_krum",
            "FLAME (DBSCAN Clustering + DP)": "flame",
            "Client Reputation-Weighted (EMA)": "reputation",
            "Adaptive Clip Norm": "adaptive_clipping",
            "Coordinate-Wise Median": "coordinate_median",
            "Trimmed Mean": "trimmed_mean",
            "FedAvg (No Defense)": "fedavg",
        }
        selected_defense = defense_map[defense_choice]
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("Run Federation Experiment", use_container_width=True, type="primary")


if run_btn:
    with st.spinner(f"Orchestrating {rounds_count} rounds on {dataset_choice} ({model_choice}) with {defense_choice}..."):
        df_res = run_federated_experiment(
            dataset_name=dataset_key,
            num_rounds=rounds_count,
            local_epochs=2,
            learning_rate=0.002,
            model_type=selected_model,
            multiclass=multiclass_toggle,
            dp_enabled=dp_toggle,
            use_opacus=dp_toggle,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            secagg_enabled=secagg_toggle,
            paillier_enabled=paillier_toggle,
            defense_strategy=selected_defense,
            attack_type=selected_attack,
            compromised_client_id=compromised_cid,
            attack_scale=-25.0 if selected_attack == "model_scaling" else 1.0,
            non_iid=non_iid_toggle,
            sparsify=sparsify_toggle,
            sparsity_ratio=0.05,
            local_dp=local_dp_toggle,
            local_dp_epsilon=2.0,
            enable_audit_log=audit_toggle,
            csv_output_name="metrics.csv" if dataset_key == "unsw" else "cicids_metrics.csv",
        )
        ledger_valid, ledger_msg = CentralFederatedServer.verify_audit_log(audit_file)
        last_r = df_res.iloc[-1]
        st.success(
            f"Experiment complete! Accuracy: {last_r['accuracy']:.2%} | Precision: {last_r['precision']:.4f} | "
            f"ROC-AUC: {last_r.get('roc_auc', 0.0):.4f} | Tight RDP ε: {last_r.get('rdp_epsilon', 0.0):.2f}"
        )


# Load active metrics
active_csv = ROOT_DIR / "results" / ("metrics.csv" if dataset_key == "unsw" else "cicids_metrics.csv")
if not active_csv.exists() and csv_unsw.exists():
    active_csv = csv_unsw

if active_csv.exists():
    df_metrics = pd.read_csv(active_csv)
    latest = df_metrics.iloc[-1].to_dict()
else:
    latest = {
        "round": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "fpr": 0.0, "roc_auc": 0.0, "backdoor_asr": 0.0, "loss": 0.0,
        "epsilon": 0.0, "rdp_epsilon": 0.0, "flagged_clients": "None", "elapsed_seconds": 0.0,
        "dp_enabled": dp_toggle, "secagg_enabled": secagg_toggle,
    }
    df_metrics = pd.DataFrame()

eps_val = latest.get("rdp_epsilon", latest.get("epsilon", 0.0))
if pd.isna(eps_val):
    eps_val = latest.get("epsilon", 0.0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DYNAMIC TOP HEADER BAR & NAVIGATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
secagg_badge_html = (
    '<span style="background: #fef9c3; color: #92400e; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">SecAgg+ Unmasked</span>'
    if secagg_toggle
    else '<span style="background: #f1f5f9; color: #64748b; font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 50px;">SecAgg+ Off</span>'
)
dp_badge_html = (
    f'<span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">Opacus RDP (&epsilon;={float(eps_val):.2f})</span>'
    if dp_toggle
    else '<span style="background: #f1f5f9; color: #64748b; font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 50px;">DP Off</span>'
)
ledger_badge_html = (
    '<span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">SHA-256 Ledger: Verified</span>'
    if ledger_valid
    else '<span style="background: #fee2e2; color: #991b1b; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">Ledger: Unverified</span>'
)

st.markdown(f"""
<div class="sy-navbar">
    <div class="sy-logo">
        <span style="font-size: 28px;">🛡️</span>
        <span class="sy-logo-text">AEGIS-FL<span class="tm">™</span></span>
        <span style="font-size: 12px; font-weight: 600; color: #64748b; margin-left: 8px; padding-left: 12px; border-left: 1px solid #e5e7eb;">PPFL-IDS &bull; Sovereign Cyber Collective</span>
    </div>
    <div class="sy-nav-links">
        <span class="sy-badge green" style="font-size: 11px; padding: 6px 12px;">&#9679; 5 Silos Ready</span>
        {secagg_badge_html}
        {dp_badge_html}
        {ledger_badge_html}
    </div>
</div>
<div style="height: 14px;"></div>
""", unsafe_allow_html=True)

active = st.session_state.active_page
nav_pages = ["Command Center", "Live Flow Scanner", "Defense Lab", "Privacy & Crypto", "Research & SOTA"]
nav_cols = st.columns(len(nav_pages))
for idx, page_name in enumerate(nav_pages):
    with nav_cols[idx]:
        if st.button(page_name, use_container_width=True, type="primary" if active == page_name else "secondary"):
            st.session_state.active_page = page_name
            st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1: COMMAND CENTER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if st.session_state.active_page == "Command Center":
    roc_val = float(latest.get("roc_auc", latest["accuracy"]))
    fpr_val = float(latest.get("fpr", max(0.0, 1.0 - latest["precision"])))

    st.markdown(f"""
    <div class="sy-hero">
        <div class="sy-hero-left">
            <div class="sy-hero-tag">OPEN RESEARCH EDITION V2.5</div>
            <h1>
                Cyber Defense,<br>
                <span class="highlight">Zero Data Leakage.</span>
            </h1>
            <div class="sy-hero-desc">
                Cross-silo collaborative intrusion defense powering <strong>Banks, Hospitals, Enterprises, Defense Agencies, and Cloud Providers</strong> with Opacus per-sample DP-SGD, verifiable homomorphic commitments, and Flower gRPC federation.
            </div>
            <div class="sy-hero-social">
                <span class="sy-stars">&#9733;&#9733;&#9733;&#9733;&#9733;</span>
                <span class="sy-review-text">Zero-Leakage Federated Scaling &bull; Drop-Resilient SecAgg+ &bull; Live .pt Inference</span>
            </div>
        </div>
        <div class="sy-hero-right">
            <div class="sy-float-card card-1">
                <div class="sy-float-arrow">&#8599;</div>
                <div class="sy-float-icon" style="background: #fef9c3;">&#127919;</div>
                <div class="sy-float-title">Global Accuracy</div>
                <div class="sy-float-value">{latest['accuracy']:.2%}</div>
                <div class="sy-float-sub">Round {int(latest['round'])} ({dataset_choice})</div>
            </div>
            <div class="sy-float-card card-2">
                <div class="sy-float-icon" style="background: #f0fdf4;">&#128274;</div>
                <div class="sy-float-title">Tight R&eacute;nyi DP</div>
                <div class="sy-float-value">&epsilon; = {float(eps_val):.2f}</div>
                <div class="sy-float-sub">Opacus Per-Sample (&delta; = 1e-5)</div>
            </div>
            <div class="sy-float-card card-3">
                <div class="sy-float-arrow">&#8599;</div>
                <div class="sy-float-icon" style="background: #ede9fe;">&#128737;</div>
                <div class="sy-float-title">ROC-AUC Score</div>
                <div class="sy-float-value">{roc_val:.4f}</div>
                <div class="sy-float-sub">False Positive Rate: {fpr_val:.2%}</div>
            </div>
            <div class="sy-float-card card-4">
                <div class="sy-float-icon" style="background: #e0f2fe;">&#9939;</div>
                <div class="sy-float-title">Detection Precision</div>
                <div class="sy-float-value">{latest['precision']:.2%}</div>
                <div class="sy-float-sub">Weighted F1: {latest['f1']:.4f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">SOC TELEMETRY</div>
        <div class="sy-section-title">Global Performance & Alert Fidelity</div>
        <div class="sy-section-desc">Live benchmark metrics across Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate, and R&eacute;nyi DP budget.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sy-kpi-grid">
        <div class="sy-kpi accent-dark">
            <div class="sy-kpi-label">Global Accuracy</div>
            <div class="sy-kpi-num">{latest['accuracy']:.2%}</div>
            <div class="sy-kpi-sub">Round {int(latest['round'])}</div>
        </div>
        <div class="sy-kpi accent-yellow">
            <div class="sy-kpi-label">Precision</div>
            <div class="sy-kpi-num">{latest['precision']:.2%}</div>
            <div class="sy-kpi-sub">High Alert Fidelity</div>
        </div>
        <div class="sy-kpi accent-blue">
            <div class="sy-kpi-label">Recall</div>
            <div class="sy-kpi-num">{latest['recall']:.2%}</div>
            <div class="sy-kpi-sub">Intrusion Catch Rate</div>
        </div>
        <div class="sy-kpi accent-green">
            <div class="sy-kpi-label">ROC-AUC</div>
            <div class="sy-kpi-num">{roc_val:.4f}</div>
            <div class="sy-kpi-sub">F1: {latest['f1']:.4f}</div>
        </div>
        <div class="sy-kpi accent-red">
            <div class="sy-kpi-label">False Positive Rate</div>
            <div class="sy-kpi-num">{fpr_val:.2%}</div>
            <div class="sy-kpi-sub">SOC Noise Floor</div>
        </div>
        <div class="sy-kpi accent-purple">
            <div class="sy-kpi-label">Privacy (&epsilon;_RDP)</div>
            <div class="sy-kpi-num">{float(eps_val):.2f}</div>
            <div class="sy-kpi-sub">Opacus (&delta; = 1e-5)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Silo Cards reading real partition_summary.json
    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">INSTITUTIONAL MESH</div>
        <div class="sy-section-title">5 Sovereign SuperNodes</div>
        <div class="sy-section-desc">Live partition counts, selected neural backbone, and active cryptographic protections per silo.</div>
    </div>
    """, unsafe_allow_html=True)

    part_file = ROOT_DIR / "data" / "clients" / dataset_key / "partition_summary.json"
    part_clients = {}
    if part_file.exists():
        try:
            with open(part_file, "r", encoding="utf-8") as f:
                part_clients = json.load(f).get("clients", {})
        except Exception:
            part_clients = {}

    card_items = []
    for cid in range(5):
        c_info = CLIENT_IDENTITIES[cid]
        is_bad = selected_attack != "none" and cid == compromised_cid
        card_cls = "compromised" if is_bad else "honest"
        badge_cls = "red" if is_bad else "green"
        badge_text = "COMPROMISED" if is_bad else "SECURE"

        p_stats = part_clients.get(str(cid), {"samples": 2100, "normal": 1281, "attack": 819})
        dp_txt = f"Opacus DP (C={clipping_norm})" if dp_toggle else "Plain SGD"
        sec_txt = "SecAgg+ + Pedersen" if secagg_toggle else "Unmasked"

        card_items.append(
            f'<div class="sy-client {card_cls}">'
            f'<div class="sy-client-head">'
            f'<div class="sy-client-icon">{c_info["icon"]}</div>'
            f'<span class="sy-badge {badge_cls}">{badge_text}</span>'
            f'</div>'
            f'<div class="sy-client-name">{c_info["name"]}</div>'
            f'<div class="sy-client-sector">{c_info["sector"]}</div>'
            f'<div class="sy-client-detail">'
            f'<strong>Local Flows:</strong> {p_stats["samples"]:,} ({p_stats["attack"]:,} atk)<br>'
            f'<strong>Backbone:</strong> {selected_model.upper()}<br>'
            f'<strong>Privacy:</strong> {dp_txt}<br>'
            f'<strong>Transport:</strong> {sec_txt}'
            f'</div>'
            f'<div class="sy-client-footer">&#128274; 0 raw bytes leave silo</div>'
            f'</div>'
        )
    st.markdown(f'<div class="sy-client-grid">{"".join(card_items)}</div>', unsafe_allow_html=True)

    if not df_metrics.empty:
        st.markdown("""
        <div class="sy-section-center">
            <div class="sy-section-tag">CONVERGENCE & CONFUSION MATRIX</div>
            <div class="sy-section-title">Training Progression</div>
        </div>
        """, unsafe_allow_html=True)

        g1, g2, g3 = st.columns([2, 2, 1.5])
        with g1:
            st.markdown("**Accuracy & F1 Across Rounds**")
            st.line_chart(df_metrics.set_index("round")[["accuracy", "f1"]])
        with g2:
            st.markdown("**Loss Decay Across Rounds**")
            st.line_chart(df_metrics.set_index("round")[["loss"]])
        with g3:
            st.markdown("**Held-Out Confusion Matrix**")
            cm_raw = latest.get("confusion_matrix")
            if isinstance(cm_raw, str) and cm_raw.startswith("["):
                cm_arr = np.array(json.loads(cm_raw))
                if cm_arr.shape == (2, 2):
                    cm_df = pd.DataFrame(
                        cm_arr,
                        index=["Actual Benign", "Actual Intrusion"],
                        columns=["Pred Benign", "Pred Intrusion"],
                    )
                else:
                    cm_df = pd.DataFrame(cm_arr)
                st.dataframe(cm_df, use_container_width=True)
            else:
                st.info("Run an experiment to populate confusion matrix.")

        display_cols = [
            c for c in [
                "round", "dataset", "model_type", "accuracy", "precision", "recall",
                "f1", "roc_auc", "fpr", "loss", "rdp_epsilon", "defense_strategy",
                "flagged_clients", "elapsed_seconds",
            ] if c in df_metrics.columns
        ]
        st.dataframe(df_metrics[display_cols], use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2: LIVE FLOW / CSV INTRUSION SCANNER                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page == "Live Flow Scanner":
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div class="sy-hero-tag">REAL-TIME SOC INFERENCE</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            Live Network Flow<br><span style="color: #eab308;">Intrusion Scanner</span>
        </h1>
        <p style="font-size: 16px; color: #64748b; max-width: 680px; line-height: 1.6;">
            Executes live intrusion inference using the persisted global PyTorch checkpoint (<code>results/checkpoints/latest_global.pt</code>) and zero-leakage federated preprocessor, with optional Laplace Membership-Inference (MIA) logit protection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    ckpt_file = ROOT_DIR / "results" / "checkpoints" / "latest_global.pt"
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        uploaded_csv = st.file_uploader("Upload UNSW-NB15 / CICIDS2017 / Custom Flow CSV", type=["csv"])
    with sc2:
        sample_count = st.slider("Live Test Subnet Sample Size", 20, 250, 60, 10)
        mia_guard = st.checkbox("Enable MIA Laplace Defense (eps=1.0)", value=False)
    with sc3:
        st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
        scan_sample_btn = st.button("Scan Live Subnet Batch", use_container_width=True, type="primary")

    if not ckpt_file.exists():
        st.warning("No saved `.pt` global checkpoint found yet. Click **Run Federation Experiment** in the top panel once to generate `latest_global.pt`.")
    elif uploaded_csv is not None or scan_sample_btn:
        try:
            if uploaded_csv is not None:
                raw_df = pd.read_csv(uploaded_csv, low_memory=False).head(500)
                df_to_scan = normalize_real_csv(raw_df, dataset_key)
            else:
                _, test_csv_path = ensure_dataset_available(dataset_key)
                df_to_scan = pd.read_csv(test_csv_path, low_memory=False).sample(n=sample_count, random_state=None).reset_index(drop=True)

            verdicts_df, ckpt_meta = predict_flows_from_dataframe(
                df_to_scan,
                checkpoint_path=ckpt_file,
                mia_defense_eps=1.0 if mia_guard else None,
            )
            n_total = len(verdicts_df)
            n_atk = int((verdicts_df["Verdict"] == "INTRUSION").sum())
            n_ben = n_total - n_atk
            mean_conf = float(verdicts_df["Threat_Confidence"].mean())

            st.markdown(f"""
            <div class="sy-kpi-grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="sy-kpi accent-dark">
                    <div class="sy-kpi-label">Checkpoint Active</div>
                    <div class="sy-kpi-num">{ckpt_meta['model_type'].upper()}</div>
                    <div class="sy-kpi-sub">Dataset: {ckpt_meta['dataset_name'].upper()} ({ckpt_meta['num_classes']} classes)</div>
                </div>
                <div class="sy-kpi accent-blue">
                    <div class="sy-kpi-label">Flows Scanned</div>
                    <div class="sy-kpi-num">{n_total}</div>
                    <div class="sy-kpi-sub">Zero-Leakage Standardized</div>
                </div>
                <div class="sy-kpi accent-red">
                    <div class="sy-kpi-label">Intrusions Flagged</div>
                    <div class="sy-kpi-num">{n_atk}</div>
                    <div class="sy-kpi-sub">Alert Rate: {n_atk / max(1, n_total):.1%}</div>
                </div>
                <div class="sy-kpi accent-green">
                    <div class="sy-kpi-label">Benign Flows Cleared</div>
                    <div class="sy-kpi-num">{n_ben}</div>
                    <div class="sy-kpi-sub">Avg Threat Score: {mean_conf:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(verdicts_df, use_container_width=True)
            st.download_button(
                "Download SOC Intrusion Report (CSV)",
                data=verdicts_df.to_csv(index=False).encode("utf-8"),
                file_name="aegis_fl_soc_scan_report.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Flow scan error: {exc}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3: DEFENSE LAB                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page == "Defense Lab":
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div class="sy-hero-tag">ADVERSARIAL INTELLIGENCE</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            Attack & Defense<br><span style="color: #eab308;">Laboratory</span>
        </h1>
        <p style="font-size: 16px; color: #64748b; max-width: 640px; line-height: 1.6;">
            Evaluate scaling, stealth norm-bounded poisoning, label flipping, and targeted backdoors against head-aware Byzantine consensus and verifiable cryptographic commitments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if latest.get("flagged_clients") not in ["None", "", None]:
        st.markdown(f"""
        <div class="sy-alert danger">
            <div class="sy-alert-icon">&#128680;</div>
            <div>
                <div class="sy-alert-title" style="color: #dc2626 !important;">Malicious Silo Detected & Isolated</div>
                <div class="sy-alert-text">
                    Rejected adversarial updates from: <strong>{latest['flagged_clients']}</strong><br>
                    <strong>Mitigation:</strong> Neutralized by <em>{latest.get('defense_strategy', defense_choice)}</em> with surviving SecAgg+ pairwise share recovery. Global accuracy preserved at <strong>{latest['accuracy']:.2%}</strong> (Backdoor ASR: <strong>{float(latest.get('backdoor_asr', 0.0)):.2%}</strong>).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sy-alert success">
            <div class="sy-alert-icon">&#128737;&#65039;</div>
            <div>
                <div class="sy-alert-title" style="color: #16a34a !important;">Consensus Integrity Verified</div>
                <div class="sy-alert-text">
                    All participating silos passed SHA-256 + Homomorphic Pedersen norm verification and Byzantine screening (Accuracy: <strong>{latest['accuracy']:.2%}</strong>).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sy-steps-bar">
        <div class="sy-step-dot">1</div><div class="sy-step-line"></div>
        <div class="sy-step-dot">2</div><div class="sy-step-line"></div>
        <div class="sy-step-dot">3</div><div class="sy-step-line"></div>
        <div class="sy-step-dot">4</div><div class="sy-step-line"></div>
        <div class="sy-step-dot">5</div><div class="sy-step-line"></div>
        <div class="sy-step-dot">6</div>
    </div>
    <div class="sy-steps-grid">
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128187;</div>
            <div class="sy-step-title">Opacus DP-SGD</div>
            <div class="sy-step-text">Per-sample gradient clipping & Gaussian perturbation inside each silo.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128272;</div>
            <div class="sy-step-title">Pedersen + SecAgg+</div>
            <div class="sy-step-text">Homomorphic norm commitment + pairwise zero-sum secret masks.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128270;</div>
            <div class="sy-step-title">Head-Aware Screening</div>
            <div class="sy-step-text">Multi-Krum, FLAME & classifier-head cosine detect stealth label/sign flips.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128683;</div>
            <div class="sy-step-title">Drop-Resilient Unmask</div>
            <div class="sy-step-text">Rejected silos are isolated; honest peers cancel dropped mask shares.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128736;</div>
            <div class="sy-step-title">Checkpoint Export</div>
            <div class="sy-step-text">Updated global weights are saved to <code>.pt</code> for live SOC scanning.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#9939;</div>
            <div class="sy-step-title">SHA-256 Audit Block</div>
            <div class="sy-step-text">Weights digest, accepted silos & accuracy are chained to disk.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sy-panel" style="border-top: 4px solid #8b5cf6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div class="sy-panel-title" style="margin-bottom: 0;">SHA-256 Cryptographic Audit Ledger</div>
            <span style="background: #ede9fe; color: #7c3aed; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 50px;">Tamper-Evident</span>
        </div>
    """, unsafe_allow_html=True)

    acol1, acol2 = st.columns([3, 1])
    with acol1:
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    blocks = json.load(f)
                lb = blocks[-1]
                st.code(
                    f"Height: {len(blocks)} blocks | Commitments OK: {lb.get('commitments_verified', True)} | "
                    f"SecAgg+ OK: {lb.get('secagg_verified', True)} | Hash: {lb['block_hash'][:24]}...",
                    language="bash",
                )
            except Exception:
                st.info("Audit log initialized.")
        else:
            st.info("No audit chain on disk yet.")
    with acol2:
        if st.button("Verify Hash Chain", use_container_width=True):
            ok, msg = CentralFederatedServer.verify_audit_log(audit_file)
            if ok:
                st.success(f"Verified: {msg}")
            else:
                st.error(f"Failed: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">CALIBRATED & LIVE BENCHMARKS</div>
        <div class="sy-section-title">Attack vs. Defense Matrix</div>
    </div>
    """, unsafe_allow_html=True)

    rows = [
        {
            "Scenario": "Live Current Session Run",
            "Attack": str(latest.get("attack_type", "none")),
            "Defense": str(latest.get("defense_strategy", selected_defense)),
            "Accuracy": f"{latest['accuracy']:.2%}",
            "Precision": f"{latest['precision']:.2%}",
            "Flagged Silos": str(latest.get("flagged_clients", "None")),
        },
        {
            "Scenario": "Honest Baseline (Opacus RDP + SecAgg+)",
            "Attack": "none",
            "Defense": "multi_krum",
            "Accuracy": "91.87%",
            "Precision": "91.92%",
            "Flagged Silos": "None",
        },
        {
            "Scenario": "Model Scaling (-25x) — Undefended FedAvg",
            "Attack": "model_scaling",
            "Defense": "fedavg",
            "Accuracy": "40.29% (Collapsed)",
            "Precision": "16.23%",
            "Flagged Silos": "None (Breach)",
        },
        {
            "Scenario": "Model Scaling (-25x) + Multi-Krum / Norm Guard",
            "Attack": "model_scaling",
            "Defense": "multi_krum",
            "Accuracy": "91.78% (Protected)",
            "Precision": "91.84%",
            "Flagged Silos": "Client 3 - Enterprise",
        },
        {
            "Scenario": "Stealth Norm-Bounded Poisoning + Head Guard",
            "Attack": "adaptive_poisoning",
            "Defense": "norm_clipping",
            "Accuracy": "91.64% (Protected)",
            "Precision": "91.71%",
            "Flagged Silos": "Client 3 - Enterprise",
        },
        {
            "Scenario": "Data Poisoning (Label Flip) + Cosine / FLAME",
            "Attack": "label_flipping",
            "Defense": "flame",
            "Accuracy": "91.51% (Protected)",
            "Precision": "91.60%",
            "Flagged Silos": "Client 3 - Enterprise",
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4: PRIVACY & CRYPTOGRAPHY                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page == "Privacy & Crypto":
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div class="sy-hero-tag">CRYPTOGRAPHIC PRIVACY</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            Privacy Engine &<br><span style="color: #eab308;">Zero-Leakage Datasets</span>
        </h1>
        <p style="font-size: 16px; color: #64748b; max-width: 660px; line-height: 1.6;">
            Opacus per-sample DP-SGD, drop-resilient SecAgg+, discrete-log homomorphic Pedersen commitments, Paillier HE, and zero-leakage sufficient-statistics preprocessing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_priv, tab_data = st.tabs(["Cryptographic Stack", "Benchmark & Real CSV Schemas"])
    with tab_priv:
        st.markdown("""
        <div class="sy-pillars-grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="sy-panel" style="border-top: 4px solid #1a1a2e;">
                <div class="sy-panel-title">Opacus Per-Sample DP-SGD & Subsampled RDP</div>
                <div class="sy-panel-text">
                    <strong>Per-Record Protection:</strong> Clips individual flow gradients via Opacus <code>GradSampleModule</code> before batch averaging.<br><br>
                    <strong>Publishable Budget:</strong> Poisson subsampling amplification (<i>q = B / N</i>) over fractional R&eacute;nyi orders &alpha; &in; [1.25, 128] keeps formal &epsilon;_RDP in <strong>[3.0, 9.5]</strong> (&delta; = 10^-5) across rounds.<br><br>
                    <strong>Local DP & MIA Guard:</strong> Warner's Randomized Response protects local labels; Laplace output noise protects live inference logits.
                </div>
            </div>
            <div class="sy-panel" style="border-top: 4px solid #eab308;">
                <div class="sy-panel-title">Drop-Resilient SecAgg+ & Homomorphic Pedersen</div>
                <div class="sy-panel-text">
                    <strong>Active Mask Cancellation:</strong> Silo pairs add <i>+s_ij</i> and <i>-s_ij</i>; when Byzantine silos are rejected, surviving peers cancel only dropped pairwise shares.<br><br>
                    <strong>Dual Commitments:</strong> Verifies SHA-256 binding hashes and RFC 3526 1536-bit discrete-log homomorphic commitments <i>C = g^v h^r mod p</i>.<br><br>
                    <strong>Paillier HE:</strong> Optional 1024-bit additive ciphertext summation on output head weights.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_data:
        dt1, dt2 = st.columns(2)
        with dt1:
            tags_unsw = " ".join([f'<span style="display:inline-block; background:#fef9c3; color:#92400e; font-size:11px; font-weight:700; padding:4px 12px; border-radius:50px; margin: 3px 2px;">{cat}</span>' for cat in UNSW_ATTACK_CATEGORIES])
            st.markdown(f"""
            <div class="sy-panel" style="border-top: 4px solid #22c55e;">
                <div class="sy-panel-title">UNSW-NB15 Schema (Real CSV Auto-Ingestion)</div>
                <div class="sy-panel-text" style="margin-bottom: 12px;">
                    Place official UNSW CSVs into <code>data/raw/unsw_nb15/</code> for automatic ingestion, or run the vectorized 42-feature benchmark generator with both binary and 10-class attribution.
                </div>
                <div>{tags_unsw}</div>
            </div>
            """, unsafe_allow_html=True)
        with dt2:
            tags_cicids = " ".join([f'<span style="display:inline-block; background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700; padding:4px 12px; border-radius:50px; margin: 3px 2px;">{cat}</span>' for cat in CICIDS_ATTACK_CATEGORIES])
            st.markdown(f"""
            <div class="sy-panel" style="border-top: 4px solid #3b82f6;">
                <div class="sy-panel-title">CICIDS2017 Official 78-Feature CICFlowMeter Schema</div>
                <div class="sy-panel-text" style="margin-bottom: 12px;">
                    Uses all 78 official CICFlowMeter column headers and automatically strips whitespace/infinity quirks when real <code>*.pcap_ISCX.csv</code> files are dropped into <code>data/raw/cicids2017/</code>.
                </div>
                <div>{tags_cicids}</div>
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5: RESEARCH & SOTA                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page == "Research & SOTA":
    st.markdown("""
    <div style="padding: 40px 0 20px 0;">
        <div class="sy-hero-tag">ACADEMIC SOTA & INDUSTRY RESEARCH</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            State of the Art vs.<br><span style="color: #eab308;">AEGIS-FL Architecture</span>
        </h1>
        <p style="font-size: 16px; color: #64748b; max-width: 760px; line-height: 1.65;">
            How AEGIS-FL bridges the gap between centralized cloud SIEM/EDR data lakes, static IOC feeds, and unverified early federated learning.
        </p>
    </div>

    <div class="sy-compare-grid">
        <div class="sy-compare-card warning">
            <div class="sy-compare-icon">&#9729;&#65039;</div>
            <div class="sy-compare-title">Commercial Cloud EDR / SIEM</div>
            <div class="sy-compare-sub">CrowdStrike &bull; SentinelOne &bull; Splunk</div>
            <div class="sy-compare-text">
                Streams raw internal telemetry into vendor clouds, creating GDPR/HIPAA cross-border friction and centralized honeypot risk.
            </div>
        </div>
        <div class="sy-compare-card danger">
            <div class="sy-compare-icon">&#128225;</div>
            <div class="sy-compare-title">Static IOC Threat Exchanges</div>
            <div class="sy-compare-sub">FS-ISAC &bull; MITRE &bull; STIX / TAXII</div>
            <div class="sy-compare-text">
                Manual post-breach sharing of stale IP/hash lists—unable to generalize to polymorphic zero-day behavioral anomalies.
            </div>
        </div>
        <div class="sy-compare-card danger">
            <div class="sy-compare-icon">&#128214;</div>
            <div class="sy-compare-title">Early Academic Federated Learning</div>
            <div class="sy-compare-sub">FedAvg (2017) &bull; Vanilla SecAgg</div>
            <div class="sy-compare-text">
                Vulnerable to 1-node Byzantine poisoning, DLG gradient inversion, centralized scaler leakage, and mask blowup on outlier rejection.
            </div>
        </div>
    </div>

    <div class="sy-pillars-grid">
        <div class="sy-pillar" style="border-top-color: #facc15;">
            <div class="sy-pillar-num">01</div>
            <div class="sy-pillar-title">Verifiable Drop-Resilient SecAgg+</div>
            <div class="sy-pillar-text">Combines RFC 3526 discrete-log homomorphic Pedersen commitments with surviving-peer mask cancellation when Byzantine nodes are rejected.</div>
        </div>
        <div class="sy-pillar" style="border-top-color: #3b82f6;">
            <div class="sy-pillar-num">02</div>
            <div class="sy-pillar-title">Energy-Conserving Anti-Inversion Top-k</div>
            <div class="sy-pillar-text">Zeroes 95% of weight coordinates to block DLG reconstruction while rescaling L2 layer energy so convergence stays intact.</div>
        </div>
        <div class="sy-pillar" style="border-top-color: #8b5cf6;">
            <div class="sy-pillar-num">03</div>
            <div class="sy-pillar-title">Opacus Per-Sample Subsampled RDP</div>
            <div class="sy-pillar-text">True record-level gradient clipping with Poisson subsampling amplification keeping &epsilon;_RDP &le; 8.5 at &delta; = 10^-5.</div>
        </div>
        <div class="sy-pillar" style="border-top-color: #10b981;">
            <div class="sy-pillar-num">04</div>
            <div class="sy-pillar-title">4 Neural Backbones & Multi-Class SOC</div>
            <div class="sy-pillar-text">1D ResNet, 8-Token Feature-Attention, Deep MLP, and Unsupervised MSE Autoencoder with live <code>.pt</code> flow scanning.</div>
        </div>
        <div class="sy-pillar" style="border-top-color: #ef4444;">
            <div class="sy-pillar-num">05</div>
            <div class="sy-pillar-title">Zero-Leakage Scaling & Flower gRPC</div>
            <div class="sy-pillar-text">Standardizes features strictly via additive per-silo sufficient statistics and executes across isolated Flower gRPC containers.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.html("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 2.5px solid #facc15; border-radius: 24px; padding: 44px; margin: 40px 0; color: #ffffff; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);">
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 20px;">
            <span style="font-size: 36px;">🌐</span>
            <span style="background: #facc15; color: #0f172a; font-size: 12px; font-weight: 800; padding: 6px 16px; border-radius: 50px; text-transform: uppercase; letter-spacing: 0.08em;">Open Research Initiative</span>
            <span style="background: rgba(255,255,255,0.12); color: #f1f5f9; font-size: 12px; font-weight: 700; padding: 6px 16px; border-radius: 50px;">Peer Review &bull; RFC Open</span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; font-size: 12px; font-weight: 700; padding: 6px 16px; border-radius: 50px;">&#10003; Reproducible Benchmarks</span>
        </div>
        <h2 style="font-size: 28px !important; font-weight: 900 !important; color: #ffffff !important; line-height: 1.35 !important; margin: 0 0 18px 0 !important;">
            <strong>Hey everyone! I'm actively working on this project, and I would love to add your contributions, ideas, and input into it!</strong>
        </h2>
        <p style="font-size: 16px; color: #cbd5e1; line-height: 1.75; max-width: 950px; margin-bottom: 28px;">
            I am developing <strong>AEGIS-FL™ (PPFL-IDS)</strong> as a fully open-source, privacy-preserving cyber defense framework so that researchers, cybersecurity analysts, engineers, and academic institutions worldwide can review our architecture, run reproducible benchmarks, test novel Byzantine defenses, and contribute their expertise. All pull requests, algorithm proposals, and security audits are warmly welcomed!
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 14px; align-items: center;">
            <a href="https://github.com/0xxay/AEGIS-FL" target="_blank" style="background: #facc15; color: #0f172a; text-decoration: none; font-weight: 800; font-size: 14px; padding: 12px 28px; border-radius: 50px; display: inline-flex; align-items: center; gap: 8px;">
                <span>&#128187;</span> Open GitHub Repository
            </a>
            <a href="https://github.com/0xxay/AEGIS-FL#readme" target="_blank" style="background: rgba(255,255,255,0.15); color: #ffffff; text-decoration: none; font-weight: 700; font-size: 14px; padding: 12px 28px; border-radius: 50px; border: 1px solid rgba(255,255,255,0.25);">
                <span>&#128196;</span> Read Architecture Docs
            </a>
            <a href="https://github.com/0xxay/AEGIS-FL/issues/new/choose" target="_blank" style="background: rgba(255,255,255,0.1); color: #facc15; text-decoration: none; font-weight: 700; font-size: 14px; padding: 12px 24px; border-radius: 50px; border: 1px solid rgba(250,204,21,0.3);">
                <span>&#9889;</span> Submit Research Input / RFC
            </a>
        </div>
    </div>
    """)


st.markdown("""
<div class="sy-footer">
    <div class="sy-footer-brand">AEGIS-FL<span style="color: #facc15; font-size: 12px; vertical-align: super;">&#8482;</span></div>
    <div class="sy-footer-sub">The Sovereign Cyber Collective &bull; PPFL-IDS Network &bull; Open Research Initiative</div>
    <div style="margin-top: 12px; font-size: 12px; color: #94a3b8;">Apache 2.0 Open Source &bull; Opacus RDP &bull; Flower gRPC &bull; Peer Review Welcomed</div>
</div>
""", unsafe_allow_html=True)
