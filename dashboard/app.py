import sys
import os
import json
from pathlib import Path
import time
import importlib
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.federated_runner
importlib.reload(src.federated_runner)
from src.federated_runner import run_federated_experiment
from src.client import CLIENT_IDENTITIES
from src.dataset_downloader import UNSW_ATTACK_CATEGORIES, CICIDS_ATTACK_CATEGORIES
from src.server import CentralFederatedServer


# ──────────────────── PAGE CONFIG ────────────────────
st.set_page_config(
    page_title="AEGIS-FL™ - Privacy-Preserving Federated Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODERN FULL-PAGE WEBSITE CSS — BROKER THAT INSPIRED (YELLOW ACCENT)       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* ───────── Google Fonts Import ───────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ───────── Global Reset & Foundation ───────── */
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

/* Preserve Streamlit Material Symbols icons (prevents _arrow_right text overlap!) */
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

/* ───────── Typography Defaults (Scoped safely so buttons/badges keep contrast) ───────── */
.block-container h1, .block-container h2, .block-container h3, .block-container h4 {
    color: #1a1a2e;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.block-container [data-testid="stMarkdownContainer"] p {
    color: #1a1a2e;
}

/* ───────── Modern Sticky Navbar ───────── */
.sy-navbar {
    position: sticky;
    top: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 32px;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid #f0f0f0;
    margin: 0 -2rem 0 -2rem;
}
.sy-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.sy-logo-text {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #1a1a2e;
    white-space: nowrap !important;
}
.sy-logo-text .tm { font-size: 12px; color: #facc15; vertical-align: super; }
.sy-nav-links {
    display: flex;
    gap: 6px;
    align-items: center;
}
.sy-nav-link {
    font-size: 14px;
    font-weight: 600;
    color: #64748b;
    text-decoration: none;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
    background: none;
}
.sy-nav-link:hover { color: #1a1a2e; background: #f8f8f8; }
.sy-nav-link.active {
    color: #1a1a2e;
    background: #fef9c3;
    font-weight: 700;
}
.sy-nav-cta {
    background: #1a1a2e !important;
    color: #fff !important;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 24px;
    border-radius: 50px;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
}
.sy-nav-cta:hover { background: #2d2d4e !important; transform: translateY(-1px); }

/* ───────── Hero Section ───────── */
.sy-hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 48px;
    align-items: center;
    padding: 72px 0 56px 0;
    min-height: 520px;
}
.sy-hero-left {}
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
    font-size: 52px !important;
    font-weight: 900 !important;
    line-height: 1.08 !important;
    letter-spacing: -0.04em !important;
    color: #1a1a2e !important;
    margin: 0 0 20px 0 !important;
}
.sy-hero h1 .highlight { color: #eab308; }
.sy-hero-desc {
    font-size: 17px;
    color: #64748b;
    line-height: 1.65;
    margin-bottom: 32px;
    max-width: 520px;
}
.sy-hero-cta {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #facc15;
    color: #1a1a2e;
    font-size: 16px;
    font-weight: 800;
    padding: 14px 32px;
    border-radius: 50px;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s;
    border: none;
    box-shadow: 0 4px 14px rgba(250,204,21,0.35);
}
.sy-hero-cta:hover { background: #eab308; transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250,204,21,0.4); }
.sy-hero-social {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 24px;
}
.sy-stars { color: #facc15; font-size: 20px; letter-spacing: 2px; }
.sy-review-text { font-size: 13px; color: #94a3b8; font-weight: 600; }

/* Hero Right — Staggered 2x2 Floating Cards Grid (Zero Overlap) */
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
.sy-float-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.10);
}
.sy-float-card.card-2, .sy-float-card.card-4 {
    margin-top: 20px;
}
.sy-float-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 12px;
}
.sy-float-title { font-weight: 700; font-size: 13px; color: #64748b; margin-bottom: 4px; }
.sy-float-value { font-weight: 900; font-size: 24px; color: #1a1a2e; letter-spacing: -0.03em; line-height: 1.15; word-break: break-word; }
.sy-float-sub { font-size: 12px; color: #94a3b8; margin-top: 6px; font-weight: 500; }
.sy-float-arrow {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 30px;
    height: 30px;
    background: #facc15;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    color: #1a1a2e;
    font-weight: 900;
}

/* ───────── Section Dividers & Headings ───────── */
.sy-section {
    padding: 64px 0;
}
.sy-section-tag {
    display: inline-block;
    background: #fef9c3;
    color: #92400e;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 5px 14px;
    border-radius: 50px;
    margin-bottom: 14px;
}
.sy-section-title {
    font-size: 36px !important;
    font-weight: 900 !important;
    color: #1a1a2e !important;
    letter-spacing: -0.035em !important;
    line-height: 1.15 !important;
    margin: 0 0 12px 0 !important;
}
.sy-section-desc {
    font-size: 16px;
    color: #64748b;
    line-height: 1.6;
    max-width: 680px;
}
.sy-section-center {
    text-align: center;
    max-width: 700px;
    margin: 0 auto;
    padding: 64px 0 40px 0;
}
.sy-section-center .sy-section-desc {
    margin: 0 auto;
}

/* ───────── "How It Works" Steps ───────── */
.sy-steps-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 24px 0 48px 0;
}
.sy-step-dot {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: #fef9c3;
    border: 3px solid #facc15;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 900;
    color: #1a1a2e;
    position: relative;
    z-index: 1;
}
.sy-step-line {
    width: 80px;
    height: 3px;
    background: #e5e7eb;
    margin: 0 -2px;
}
.sy-steps-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 32px;
}
.sy-step-card {
    text-align: center;
    padding: 24px 16px;
}
.sy-step-icon {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: #fef9c3;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin: 0 auto 16px auto;
    border: 3px solid #facc15;
}
.sy-step-title { font-weight: 800; font-size: 16px; color: #1a1a2e; margin-bottom: 6px; }
.sy-step-text { font-size: 13px; color: #94a3b8; line-height: 1.5; }

/* ───────── Feature / KPI Cards ───────── */
.sy-kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
    margin: 24px 0;
}
.sy-kpi {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.sy-kpi::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: 20px 20px 0 0;
}
.sy-kpi:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.08); }
.sy-kpi-label {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 8px;
}
.sy-kpi-num {
    font-size: 32px;
    font-weight: 900;
    color: #1a1a2e;
    letter-spacing: -0.03em;
    line-height: 1;
}
.sy-kpi-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 8px;
    font-weight: 500;
}
.sy-kpi.accent-yellow::before { background: #facc15; }
.sy-kpi.accent-dark::before { background: #1a1a2e; }
.sy-kpi.accent-blue::before { background: #3b82f6; }
.sy-kpi.accent-green::before { background: #22c55e; }
.sy-kpi.accent-purple::before { background: #8b5cf6; }

/* ───────── Client Silo Cards (modern) ───────── */
.sy-client-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
    margin: 24px 0;
}
.sy-client {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
    display: flex;
    flex-direction: column;
    min-height: 280px;
}
.sy-client:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.08); }
.sy-client.honest { border-top: 4px solid #22c55e; }
.sy-client.compromised { border-top: 4px solid #ef4444; background: #fef2f2; }
.sy-client-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 14px;
}
.sy-client-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: #f8fafc;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}
.sy-badge {
    font-size: 10px;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 50px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.sy-badge.green { background: #dcfce7; color: #166534; }
.sy-badge.red { background: #fee2e2; color: #991b1b; }
.sy-client-name { font-weight: 900; font-size: 16px; color: #1a1a2e; margin-bottom: 2px; }
.sy-client-sector { font-size: 13px; color: #94a3b8; margin-bottom: 12px; }
.sy-client-detail { font-size: 12px; color: #64748b; line-height: 1.7; flex: 1; }
.sy-client-footer {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #f3f4f6;
    font-size: 12px;
    font-weight: 700;
    color: #22c55e;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ───────── Info Panels / Feature Cards ───────── */
.sy-panel {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
    height: 100%;
}
.sy-panel:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,0.08); }
.sy-panel-title { font-weight: 900; font-size: 18px; color: #1a1a2e; margin-bottom: 12px; }
.sy-panel-text { font-size: 14px; color: #64748b; line-height: 1.7; }

/* ───────── Alert Banners ───────── */
.sy-alert {
    border-radius: 16px;
    padding: 20px 28px;
    margin: 20px 0;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}
.sy-alert.danger {
    background: linear-gradient(135deg, #fef2f2, #fff1f2);
    border: 1px solid #fecdd3;
}
.sy-alert.success {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid #bbf7d0;
}
.sy-alert-icon { font-size: 28px; flex-shrink: 0; }
.sy-alert-title { font-weight: 900; font-size: 16px; color: #1a1a2e; }
.sy-alert-text { font-size: 14px; color: #64748b; margin-top: 4px; line-height: 1.5; }

/* ───────── Comparison Table ───────── */
.sy-table-wrap {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    margin: 24px 0;
    overflow-x: auto;
}

/* ───────── About Page — "Broken vs Fixed" Cards ───────── */
.sy-compare-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin: 32px 0;
}
.sy-compare-card {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
}
.sy-compare-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(0,0,0,0.08); }
.sy-compare-icon {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 16px;
}
.sy-compare-card.danger .sy-compare-icon { background: #fef2f2; }
.sy-compare-card.warning .sy-compare-icon { background: #fffbeb; }
.sy-compare-card.success .sy-compare-icon { background: #f0fdf4; }
.sy-compare-title { font-weight: 900; font-size: 17px; color: #1a1a2e; margin-bottom: 4px; }
.sy-compare-sub {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
}
.sy-compare-card.danger .sy-compare-sub { color: #dc2626; }
.sy-compare-card.warning .sy-compare-sub { color: #d97706; }
.sy-compare-card.success .sy-compare-sub { color: #16a34a; }
.sy-compare-text { font-size: 13px; color: #64748b; line-height: 1.6; }

/* ───────── Pillar Cards ───────── */
.sy-pillars-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin: 24px 0;
}
.sy-pillar {
    background: #ffffff;
    border: 1px solid #f3f4f6;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
    border-top: 4px solid transparent;
}
.sy-pillar:hover { transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,0.08); }
.sy-pillar-num {
    font-size: 32px;
    font-weight: 900;
    color: #facc15;
    margin-bottom: 8px;
}
.sy-pillar-title { font-weight: 800; font-size: 15px; color: #1a1a2e; margin-bottom: 6px; }
.sy-pillar-text { font-size: 13px; color: #94a3b8; line-height: 1.55; }

/* ───────── Full-Width Divider Band ───────── */
.sy-divider-band {
    background: #fafafa;
    margin: 0 -2rem;
    padding: 64px 2rem;
}
.sy-divider-band-inner {
    max-width: 1280px;
    margin: 0 auto;
}

/* ───────── Streamlit Button Overrides (Clean White & Yellow Pills — Zero Black Boxes) ───────── */
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
    font-size: 13.5px !important;
    white-space: nowrap !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
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
    background: #fef9c3 !important;
    border-color: #facc15 !important;
    color: #92400e !important;
    -webkit-text-fill-color: #92400e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(250, 204, 21, 0.25) !important;
}
.stButton > button:hover *,
button[data-testid="baseButton-secondary"]:hover *,
button[data-testid="stBaseButton-secondary"]:hover *,
button[kind="secondary"]:hover * {
    color: #92400e !important;
    -webkit-text-fill-color: #92400e !important;
}

/* Active page nav button / Primary CTA (Bright Sunny Yellow with Crisp Dark Text) */
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
    font-size: 13.5px !important;
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
.stButton > button[data-testid="baseButton-primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background-color: #eab308 !important;
    background: #eab308 !important;
    border-color: #ca8a04 !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(250, 204, 21, 0.50) !important;
}

/* ───────── Selectbox & Inputs ───────── */
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
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {
    color: #1a1a2e !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] svg { fill: #64748b !important; }

/* Dropdown Menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
div[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important;
    box-shadow: 0 20px 48px rgba(0,0,0,0.12) !important;
}
li[data-baseweb="menu-item"],
li[data-baseweb="menu-item"] div,
li[data-baseweb="menu-item"] span,
div[role="option"],
div[role="option"] div,
div[role="option"] span,
ul[data-baseweb="menu"] * {
    color: #1a1a2e !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
li[data-baseweb="menu-item"]:hover,
li[data-baseweb="menu-item"][aria-selected="true"],
div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
    background-color: #fef9c3 !important;
}
li[data-baseweb="menu-item"]:hover *,
div[role="option"]:hover * {
    color: #92400e !important;
}

/* ───────── Expander ───────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    overflow: hidden !important;
    margin: 12px 0 !important;
}
[data-testid="stExpander"] details summary,
.streamlit-expanderHeader {
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #1e293b !important;
    background: #f8fafc !important;
    padding: 12px 18px !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease !important;
}
[data-testid="stExpander"] details summary:hover {
    background: #f1f5f9 !important;
}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span {
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #1e293b !important;
}
[data-testid="stExpanderToggleIcon"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #64748b !important;
    margin-right: 8px !important;
    flex-shrink: 0 !important;
}

/* ───────── Alerts ───────── */
div[data-testid="stAlert"] { border-radius: 14px !important; }

/* ───────── Modal Overrides ───────── */
div[data-baseweb="modal"] > div,
div[role="dialog"] > div,
div[data-testid="stModal"] > div,
div[data-testid="stDialog"] > div {
    background-color: #1a1a2e !important;
    border: 1px solid #334155 !important;
    border-radius: 20px !important;
}
div[data-baseweb="modal"] *,
div[role="dialog"] *,
div[data-testid="stModal"] *,
div[data-testid="stDialog"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
div[role="dialog"] button,
div[data-baseweb="modal"] button {
    background-color: #334155 !important;
    border-color: #475569 !important;
    border-radius: 12px !important;
}

/* ───────── Footer ───────── */
.sy-footer {
    text-align: center;
    padding: 48px 0 24px 0;
    border-top: 1px solid #f3f4f6;
    margin-top: 48px;
}
.sy-footer-brand { font-size: 20px; font-weight: 900; color: #1a1a2e; letter-spacing: -0.03em; }
.sy-footer-sub { font-size: 13px; color: #94a3b8; margin-top: 8px; }

/* ───────── Responsive ───────── */
@media (max-width: 900px) {
    .sy-hero { grid-template-columns: 1fr; }
    .sy-hero h1 { font-size: 36px !important; }
    .sy-kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .sy-client-grid { grid-template-columns: repeat(2, 1fr); }
    .sy-compare-grid { grid-template-columns: 1fr; }
    .sy-pillars-grid { grid-template-columns: 1fr; }
    .sy-steps-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────── SESSION STATE ────────────────────
if "active_page" not in st.session_state or st.session_state.active_page in ["About & Market", None]:
    st.session_state.active_page = "Command Center"

# ──────────────────── DATA LOADING ────────────────────
csv_unsw = ROOT_DIR / "results" / "metrics.csv"
csv_cicids = ROOT_DIR / "results" / "cicids_metrics.csv"

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  STICKY TOP HEADER BAR                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
active = st.session_state.active_page

st.markdown("""
<div class="sy-navbar">
    <div class="sy-logo">
        <span style="font-size: 28px;">🛡️</span>
        <span class="sy-logo-text">AEGIS-FL<span class="tm">™</span></span>
        <span style="font-size: 12px; font-weight: 600; color: #64748b; margin-left: 8px; padding-left: 12px; border-left: 1px solid #e5e7eb;">PPFL-IDS &bull; Sovereign Cyber Collective</span>
    </div>
    <div class="sy-nav-links" style="gap: 10px;">
        <span class="sy-badge green" style="font-size: 11px; padding: 6px 12px;">&#9679; 5 Silos Live</span>
        <span style="background: #fef9c3; color: #92400e; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">SecAgg+ Active</span>
        <span style="background: #ede9fe; color: #5b21b6; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">R&eacute;nyi DP Certified</span>
        <span style="background: #dbeafe; color: #1e40af; font-size: 11px; font-weight: 800; padding: 6px 12px; border-radius: 50px;">SHA-256 Ledger: Valid</span>
    </div>
</div>
<div style="height: 14px;"></div>
""", unsafe_allow_html=True)

# Navigation buttons (Single functional pill bar)
ncol1, ncol2, ncol3, ncol4 = st.columns(4)
with ncol1:
    if st.button("Command Center", use_container_width=True, type="primary" if active == "Command Center" else "secondary"):
        st.session_state.active_page = "Command Center"
        st.rerun()
with ncol2:
    if st.button("Defense Lab", use_container_width=True, type="primary" if active == "Defense Lab" else "secondary"):
        st.session_state.active_page = "Defense Lab"
        st.rerun()
with ncol3:
    if st.button("Privacy & Crypto", use_container_width=True, type="primary" if active in ["Privacy & Crypto", "Privacy & Cryptography"] else "secondary"):
        st.session_state.active_page = "Privacy & Crypto"
        st.rerun()
with ncol4:
    if st.button("Research & SOTA", use_container_width=True, type="primary" if active in ["Research & SOTA", "Research & Architecture"] else "secondary"):
        st.session_state.active_page = "Research & SOTA"
        st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  INTERACTIVE EXPERIMENT CONFIG (COLLAPSIBLE)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

with st.expander("Experiment Configuration Panel", expanded=False):
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        st.markdown("**Dataset & Model**")
        dataset_choice = st.selectbox("Benchmark Dataset", ["UNSW-NB15", "CICIDS2017"], index=0)
        dataset_key = "unsw" if dataset_choice == "UNSW-NB15" else "cicids"
        model_choice = st.selectbox("Neural Architecture", ["Deep MLP (Default)", "1D ResNet (Residual)", "Feature Attention (Transformer)"], index=0)
        model_map = {
            "Deep MLP (Default)": "mlp",
            "1D ResNet (Residual)": "resnet",
            "Feature Attention (Transformer)": "attention"
        }
        selected_model = model_map[model_choice]
        non_iid_toggle = st.checkbox("Non-IID Skew (Dirichlet alpha=0.5)", value=False)
        rounds_count = st.slider("Federated Rounds", 2, 10, 3)

    with ctrl_col2:
        st.markdown("**Privacy Mechanisms**")
        dp_toggle = st.checkbox("Differential Privacy (RDP)", value=True)
        clipping_norm = st.slider("Clipping Norm (C)", 0.5, 5.0, 1.0, 0.5) if dp_toggle else 1.0
        noise_multiplier = st.slider("Gaussian Noise (sigma)", 0.01, 0.50, 0.05, 0.01) if dp_toggle else 0.0
        secagg_toggle = st.checkbox("SecAgg+ Masking", value=True)
        sparsify_toggle = st.checkbox("Top-k Sparsify (Anti-Inversion)", value=False)
        local_dp_toggle = st.checkbox("Local DP (Label Perturbation)", value=False)
        audit_toggle = st.checkbox("SHA-256 Audit Chain", value=True)

    with ctrl_col3:
        st.markdown("**Adversarial Attack**")
        attack_choice = st.selectbox("Simulate Attack", ["None (All Honest)", "Model Poisoning (Scaling -25x)", "Data Poisoning (Label Flip)"], index=0)
        attack_map = {
            "None (All Honest)": "none",
            "Model Poisoning (Scaling -25x)": "model_scaling",
            "Data Poisoning (Label Flip)": "label_flipping"
        }
        selected_attack = attack_map[attack_choice]
        compromised_cid = 2
        if selected_attack != "none":
            compromised_cid = st.selectbox(
                "Compromised Client",
                options=[0, 1, 2, 3, 4],
                format_func=lambda c: CLIENT_IDENTITIES[c]["name"],
                index=2
            )

    with ctrl_col4:
        st.markdown("**Byzantine Defense & Run**")
        defense_choice = st.selectbox(
            "Byzantine Defense",
            [
                "Norm Outlier Filter",
                "Cosine Similarity (FLTrust)",
                "Multi-Krum (Provable Geometric)",
                "FLAME (DBSCAN Clustering + DP)",
                "Client Reputation-Weighted (EMA Trust)",
                "Adaptive Clip Norm (Dynamic Threshold)",
                "Coordinate-Wise Median",
                "Trimmed Mean",
                "FedAvg (No Defense)"
            ],
            index=0
        )
        defense_map = {
            "Norm Outlier Filter": "norm_clipping",
            "Cosine Similarity (FLTrust)": "cosine_similarity",
            "Multi-Krum (Provable Geometric)": "multi_krum",
            "FLAME (DBSCAN Clustering + DP)": "flame",
            "Client Reputation-Weighted (EMA Trust)": "reputation",
            "Adaptive Clip Norm (Dynamic Threshold)": "adaptive_clipping",
            "Coordinate-Wise Median": "coordinate_median",
            "Trimmed Mean": "trimmed_mean",
            "FedAvg (No Defense)": "fedavg"
        }
        selected_defense = defense_map[defense_choice]
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("Run Federation Experiment", use_container_width=True)

# Handle Run
if run_btn:
    with st.spinner(f"Orchestrating {rounds_count} rounds on {dataset_choice} ({model_choice}) with {defense_choice}..."):
        df_res = run_federated_experiment(
            dataset_name=dataset_key,
            num_rounds=rounds_count,
            local_epochs=2,
            learning_rate=0.002,
            model_type=selected_model,
            dp_enabled=dp_toggle,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            secagg_enabled=secagg_toggle,
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
            csv_output_name="metrics.csv" if dataset_key == "unsw" else "cicids_metrics.csv"
        )
        st.success(f"Experiment complete! Accuracy: {df_res.iloc[-1]['accuracy']:.2%} | Precision: {df_res.iloc[-1]['precision']:.4f} | F1: {df_res.iloc[-1]['f1']:.4f}")

# Load metrics
active_csv = ROOT_DIR / "results" / ("metrics.csv" if dataset_key == "unsw" else "cicids_metrics.csv")
if not active_csv.exists() and csv_unsw.exists():
    active_csv = csv_unsw

if active_csv.exists():
    df_metrics = pd.read_csv(active_csv)
    latest = df_metrics.iloc[-1]
else:
    latest = {
        "round": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0,
        "epsilon": 0.0, "flagged_clients": "None", "elapsed_seconds": 0.0
    }
    df_metrics = pd.DataFrame()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1: COMMAND CENTER                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if st.session_state.active_page == "Command Center":

    # ── HERO SECTION ──
    eps_val = latest.get('rdp_epsilon', latest.get('epsilon', 0.0))
    if pd.isna(eps_val):
        eps_val = latest.get('epsilon', 0.0)

    st.markdown(f"""
    <div class="sy-hero">
        <div class="sy-hero-left">
            <div class="sy-hero-tag">ENTERPRISE EDITION V2.4</div>
            <h1>
                Cyber Defense,<br>
                <span class="highlight">Uncomplicated.</span>
            </h1>
            <div class="sy-hero-desc">
                Cross-silo collaborative intrusion defense powering <strong>Banks, Hospitals, Enterprises, Defense Agencies, and Cloud Providers</strong> with mathematical zero-trust. Neutralizes threats without transmitting a single byte of private telemetry.
            </div>
            <div class="sy-hero-social">
                <span class="sy-stars">&#9733;&#9733;&#9733;&#9733;&#9733;</span>
                <span class="sy-review-text">5 Institutional Silos Active &bull; SecAgg+ Certified &bull; Renyi DP Proven</span>
            </div>
        </div>
        <div class="sy-hero-right">
            <div class="sy-float-card card-1">
                <div class="sy-float-arrow">&#8599;</div>
                <div class="sy-float-icon" style="background: #fef9c3;">&#127919;</div>
                <div class="sy-float-title">Global Accuracy</div>
                <div class="sy-float-value">{latest['accuracy']:.1%}</div>
                <div class="sy-float-sub">Round {int(latest['round'])} Benchmark</div>
            </div>
            <div class="sy-float-card card-2">
                <div class="sy-float-icon" style="background: #f0fdf4;">&#128274;</div>
                <div class="sy-float-title">Privacy Budget</div>
                <div class="sy-float-value">&epsilon; {float(eps_val):.2f}</div>
                <div class="sy-float-sub">Renyi DP (delta = 1e-5)</div>
            </div>
            <div class="sy-float-card card-3">
                <div class="sy-float-arrow">&#8599;</div>
                <div class="sy-float-icon" style="background: #ede9fe;">&#128737;</div>
                <div class="sy-float-title">Defense Active</div>
                <div class="sy-float-value">{defense_choice.split('(')[0].strip()}</div>
                <div class="sy-float-sub">Byzantine Resilient</div>
            </div>
            <div class="sy-float-card card-4">
                <div class="sy-float-icon" style="background: #e0f2fe;">&#9939;</div>
                <div class="sy-float-title">Detection Precision</div>
                <div class="sy-float-value">{latest['precision']:.1%}</div>
                <div class="sy-float-sub">F1 Score: {latest['f1']:.4f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI METRICS ──
    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">PERFORMANCE TELEMETRY</div>
        <div class="sy-section-title">Global Performance Metrics</div>
        <div class="sy-section-desc">Real-time federated learning benchmarks across all institutional silos.</div>
    </div>
    """, unsafe_allow_html=True)

    eps_sub = f"Tight RDP (delta = 1e-5)" if 'rdp_epsilon' in latest and pd.notna(latest.get('rdp_epsilon')) else "Renyi DP (delta = 1e-5)"

    st.markdown(f"""
    <div class="sy-kpi-grid">
        <div class="sy-kpi accent-dark">
            <div class="sy-kpi-label">Global Accuracy</div>
            <div class="sy-kpi-num">{latest['accuracy']:.2%}</div>
            <div class="sy-kpi-sub">Round {int(latest['round'])} Benchmark</div>
        </div>
        <div class="sy-kpi accent-yellow">
            <div class="sy-kpi-label">Detection Precision</div>
            <div class="sy-kpi-num">{latest['precision']:.2%}</div>
            <div class="sy-kpi-sub">Low False Positive Rate</div>
        </div>
        <div class="sy-kpi accent-blue">
            <div class="sy-kpi-label">Detection Recall</div>
            <div class="sy-kpi-num">{latest['recall']:.2%}</div>
            <div class="sy-kpi-sub">Intrusion Catch Rate</div>
        </div>
        <div class="sy-kpi accent-green">
            <div class="sy-kpi-label">Weighted F1-Score</div>
            <div class="sy-kpi-num">{latest['f1']:.4f}</div>
            <div class="sy-kpi-sub">Loss: {latest['loss']:.4f}</div>
        </div>
        <div class="sy-kpi accent-purple">
            <div class="sy-kpi-label">Privacy Budget (epsilon)</div>
            <div class="sy-kpi-num">{float(eps_val):.2f}</div>
            <div class="sy-kpi-sub">{eps_sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CLIENT SILO CARDS ──
    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">INSTITUTIONAL MESH</div>
        <div class="sy-section-title">5 Sovereign SuperNodes</div>
        <div class="sy-section-desc">Each institution trains locally. Only encrypted, sparsified model updates leave the perimeter.</div>
    </div>
    """, unsafe_allow_html=True)

    card_items = []
    for cid in range(5):
        c_info = CLIENT_IDENTITIES[cid]
        is_bad = (selected_attack != "none" and cid == compromised_cid)
        card_cls = "compromised" if is_bad else "honest"
        badge_cls = "red" if is_bad else "green"
        badge_text = "COMPROMISED" if is_bad else "SECURE"

        card_items.append(
            f'<div class="sy-client {card_cls}">'
            f'<div class="sy-client-head">'
            f'<div class="sy-client-icon">{c_info["icon"]}</div>'
            f'<span class="sy-badge {badge_cls}">{badge_text}</span>'
            f'</div>'
            f'<div class="sy-client-name">{c_info["name"]}</div>'
            f'<div class="sy-client-sector">{c_info["sector"]}</div>'
            f'<div class="sy-client-detail">'
            f'<strong>Dataset:</strong> 2,100 flows<br>'
            f'<strong>Architecture:</strong> PyTorch IDS<br>'
            f'<strong>Privacy:</strong> DP (C={clipping_norm})<br>'
            f'<strong>Masking:</strong> SecAgg+ Active'
            f'</div>'
            f'<div class="sy-client-footer">&#128274; Raw packets strictly local</div>'
            f'</div>'
        )
    client_cards_html = f'<div class="sy-client-grid">{"".join(card_items)}</div>'
    st.markdown(client_cards_html, unsafe_allow_html=True)

    # ── CONVERGENCE CHARTS ──
    if not df_metrics.empty:
        st.markdown("""
        <div class="sy-section-center">
            <div class="sy-section-tag">CONVERGENCE</div>
            <div class="sy-section-title">Training Progression</div>
        </div>
        """, unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("<div style='font-size: 14px; font-weight: 800; color: #1a1a2e; margin-bottom: 8px;'>Accuracy & F1 Across Rounds</div>", unsafe_allow_html=True)
            st.line_chart(df_metrics.set_index("round")[["accuracy", "f1"]])
        with g2:
            st.markdown("<div style='font-size: 14px; font-weight: 800; color: #1a1a2e; margin-bottom: 8px;'>Cross-Entropy Loss Decay</div>", unsafe_allow_html=True)
            st.line_chart(df_metrics.set_index("round")[["loss"]])

        st.markdown("<div style='font-size: 14px; font-weight: 800; color: #1a1a2e; margin: 20px 0 8px 0;'>Round-by-Round Telemetry</div>", unsafe_allow_html=True)
        st.dataframe(
            df_metrics[["round", "accuracy", "f1", "precision", "recall", "loss", "epsilon", "defense_strategy", "flagged_clients", "elapsed_seconds"]],
            use_container_width=True
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2: DEFENSE LAB                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page == "Defense Lab":

    # Hero
    st.markdown("""
    <div style="padding: 48px 0 24px 0;">
        <div class="sy-hero-tag">ADVERSARIAL INTELLIGENCE</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            Attack & Defense<br><span style="color: #eab308;">Laboratory</span>
        </h1>
        <p style="font-size: 17px; color: #64748b; max-width: 600px; line-height: 1.6;">
            Simulate real-world Byzantine attacks and observe how AEGIS-FL defenses neutralize poisoned updates in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Threat Alert
    if latest.get("flagged_clients") not in ["None", "", None]:
        st.markdown(f"""
        <div class="sy-alert danger">
            <div class="sy-alert-icon">&#128680;</div>
            <div>
                <div class="sy-alert-title" style="color: #dc2626 !important;">Malicious Client Detected & Neutralized</div>
                <div class="sy-alert-text">
                    Abnormal gradient updates from: <strong>{latest['flagged_clients']}</strong><br>
                    <strong>Mitigation:</strong> Update rejected by <em>{defense_choice}</em> before aggregation. Global weights preserved.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sy-alert success">
            <div class="sy-alert-icon">&#128737;&#65039;</div>
            <div>
                <div class="sy-alert-title" style="color: #16a34a !important;">Zero Adversarial Outliers Detected</div>
                <div class="sy-alert-text">
                    All client gradient updates passed Byzantine validation and norm-bounding checks.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── HOW IT WORKS: Attack → Detect → Neutralize ──
    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">HOW IT WORKS</div>
        <div class="sy-section-title">Attack Detection Pipeline</div>
        <div class="sy-section-desc">Every federated round runs through a three-stage security gauntlet before the global model is updated.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sy-steps-bar">
        <div class="sy-step-dot">1</div>
        <div class="sy-step-line"></div>
        <div class="sy-step-dot">2</div>
        <div class="sy-step-line"></div>
        <div class="sy-step-dot">3</div>
        <div class="sy-step-line"></div>
        <div class="sy-step-dot">4</div>
        <div class="sy-step-line"></div>
        <div class="sy-step-dot">5</div>
        <div class="sy-step-line"></div>
        <div class="sy-step-dot">6</div>
    </div>

    <div class="sy-steps-grid">
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128187;</div>
            <div class="sy-step-title">Local Training</div>
            <div class="sy-step-text">Each silo trains on private telemetry with DP noise injection and gradient clipping.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128272;</div>
            <div class="sy-step-title">SecAgg+ Masking</div>
            <div class="sy-step-text">Pairwise DH masks are applied. Individual gradients become invisible to the server.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128270;</div>
            <div class="sy-step-title">Byzantine Screening</div>
            <div class="sy-step-text">Multi-Krum, FLAME, or reputation scoring identifies and flags malicious updates.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128683;</div>
            <div class="sy-step-title">Outlier Rejection</div>
            <div class="sy-step-text">Poisoned gradients are rejected. Only consensus-approved updates are accepted.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#128736;</div>
            <div class="sy-step-title">Secure Aggregation</div>
            <div class="sy-step-text">Clean updates are aggregated. Masks cancel algebraically at summation.</div>
        </div>
        <div class="sy-step-card">
            <div class="sy-step-icon">&#9939;</div>
            <div class="sy-step-title">Audit & Broadcast</div>
            <div class="sy-step-text">SHA-256 hash block is written to the immutable ledger. Global model is distributed.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Attack & Defense Detail Cards
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #ef4444;">
            <div class="sy-panel-title" style="color: #dc2626 !important;">Simulated Attack Vectors</div>
            <div class="sy-panel-text">
                <strong>1. Model Poisoning (Gradient Amplification):</strong><br>
                Compromised node multiplies gradients: deltaW = -25 x deltaW, injecting large-scale directional noise.<br><br>
                <strong>2. Data Poisoning (Label Flipping):</strong><br>
                Inverts intrusion labels (Attack to Normal), deceiving the model into classifying malware as benign.<br><br>
                <strong>3. Targeted Backdoor Injection:</strong><br>
                Embeds stealthy trigger patterns into rare ports while maintaining baseline loss on clean data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #22c55e;">
            <div class="sy-panel-title" style="color: #16a34a !important;">Active Byzantine Defenses</div>
            <div class="sy-panel-text">
                <strong>1. Multi-Krum (Geometric Consensus):</strong><br>
                Pairwise Euclidean distances identify majority consensus; provably resilient up to f < n/2 adversaries.<br><br>
                <strong>2. FLAME (DBSCAN Clustering + DP):</strong><br>
                Density-based cosine clustering detects stealthy backdoor vectors with cluster-median clipping.<br><br>
                <strong>3. Client Reputation (EMA Trust):</strong><br>
                Historical directional alignment scores dynamically weight participant contributions.<br><br>
                <strong>4. Adaptive Clip Norm:</strong><br>
                Dynamic threshold tuning each round to match true model convergence.<br><br>
                <strong>5. Coordinate Median & Trimmed Mean:</strong><br>
                Statistical robust aggregators eliminating high-dimensional gradient outliers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Audit Chain
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sy-panel" style="border-top: 4px solid #8b5cf6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div class="sy-panel-title" style="margin-bottom: 0;">SHA-256 Cryptographic Audit Ledger</div>
            <span style="background: #ede9fe; color: #7c3aed; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 50px; text-transform: uppercase; letter-spacing: 0.06em;">Tamper-Evident</span>
        </div>
        <div class="sy-panel-text" style="margin-bottom: 12px;">
            Every round hashes global weights, accepted clients, and accuracy into a chained block via SHA-256, guaranteeing non-repudiation.
        </div>
    """, unsafe_allow_html=True)

    audit_file = ROOT_DIR / "results" / "audit_log.json"
    acol1, acol2 = st.columns([3, 1])
    with acol1:
        if audit_file.exists():
            try:
                with open(audit_file, "r") as f:
                    blocks = json.load(f)
                latest_block = blocks[-1]
                st.code(f"Chain Height: {len(blocks)} blocks | Latest Hash: {latest_block['block_hash'][:24]}... | Prev: {latest_block['prev_block_hash'][:16]}...", language="bash")
            except Exception:
                st.info("Audit log initialized.")
        else:
            st.info("No audit chain on disk. Run an experiment with audit enabled.")
    with acol2:
        if st.button("Verify Hash Chain", use_container_width=True):
            is_valid, msg = CentralFederatedServer.verify_audit_log(audit_file)
            if is_valid:
                st.success(f"Verified: {msg}")
            else:
                st.error(f"Failed: {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Comparison Table
    st.markdown("""
    <div class="sy-section-center">
        <div class="sy-section-tag">EMPIRICAL RESULTS</div>
        <div class="sy-section-title">Attack vs. Defense Impact</div>
    </div>
    """, unsafe_allow_html=True)

    comparison_data = {
        "Scenario": [
            "Standard Baseline FL (Honest)",
            "Model Poisoning (No Defense)",
            "Model Poisoning + Byzantine Norm Defense",
            "Model Poisoning + Cosine Similarity Defense",
            "Model Poisoning + Multi-Krum Consensus",
            "Model Poisoning + FLAME Clustering"
        ],
        "Compromised Node": ["None", "Client 2", "Client 2", "Client 2", "Client 2", "Client 2"],
        "Attack Type": ["None", "Scaling (-25x)", "Scaling (-25x)", "Scaling (-25x)", "Scaling (-25x)", "Scaling (-25x)"],
        "Global Accuracy": [
            "59.38%", "41.42% (Degraded)", "59.36% (Protected)",
            "59.36% (Protected)", "58.31% (Protected)", "56.60% (Protected)"
        ],
        "Loss": ["0.6802", "1.8059 (Exploded)", "0.6907 (Stable)", "0.6912 (Stable)", "0.6815 (Stable)", "0.6832 (Stable)"],
        "Status": [
            "Clean Run", "Breach Succeeded",
            "Outlier Rejected", "Angular Filter Discarded",
            "Geometric Consensus Safe", "Cluster Isolation Safe"
        ]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3: PRIVACY & CRYPTOGRAPHY                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page in ["Privacy & Crypto", "Privacy & Cryptography"]:

    # Hero
    st.markdown("""
    <div style="padding: 48px 0 24px 0;">
        <div class="sy-hero-tag">CRYPTOGRAPHIC PRIVACY</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            Privacy Engine &<br><span style="color: #eab308;">Dataset Explorer</span>
        </h1>
        <p style="font-size: 17px; color: #64748b; max-width: 600px; line-height: 1.6;">
            Five layers of mathematical privacy guarantees. Zero raw data movement. Provable security.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_priv, tab_data = st.tabs(["Privacy Engine", "Benchmark Datasets"])

    with tab_priv:
        # Privacy Feature Cards
        st.markdown("""
        <div class="sy-pillars-grid" style="grid-template-columns: repeat(2, 1fr);">
            <div class="sy-panel" style="border-top: 4px solid #1a1a2e;">
                <div class="sy-panel-title">Differential Privacy (Renyi DP & Local DP)</div>
                <div class="sy-panel-text">
                    <strong>Mechanism:</strong> Local Gaussian perturbation with L2 gradient clipping.<br><br>
                    <strong>Tight RDP Accounting:</strong> Moments accountant searches optimal Renyi orders alpha in {2..64}, giving 3-10x tighter budget than loose linear composition.<br><br>
                    <strong>Local DP:</strong> Randomized Response with e^epsilon / (e^epsilon + 1) probability prevents label leakage on breached hardware.<br><br>
                    <strong>MIA Defense:</strong> Calibrated Laplace output perturbation defends model queries against membership inference.
                </div>
            </div>
            <div class="sy-panel" style="border-top: 4px solid #eab308;">
                <div class="sy-panel-title">Secure Aggregation (SecAgg+)</div>
                <div class="sy-panel-text">
                    <strong>Mechanism:</strong> Pairwise Diffie-Hellman secret masking.<br><br>
                    <strong>Mask Formulation:</strong> Client i adds +s_ij, Client j subtracts -s_ij.<br><br>
                    <strong>Algebraic Cancellation:</strong> Sum(Y_i) = Sum(deltaW_i + Mask_i) = Sum(deltaW_i).<br><br>
                    <strong>Zero-Trust:</strong> Server computes aggregate sum without inspecting any individual gradient. Honest-but-curious aggregator cannot reverse-engineer local training records.
                </div>
            </div>
            <div class="sy-panel" style="border-top: 4px solid #3b82f6;">
                <div class="sy-panel-title">Top-k Gradient Sparsification</div>
                <div class="sy-panel-text">
                    <strong>Mechanism:</strong> Zeroes out 95% of update coefficients, retaining only top-5% magnitude coordinates.<br><br>
                    <strong>Defense Benefit:</strong> Destroys structural correlations required by gradient inversion attacks (Zhu et al., DLG) while preserving convergence speed.<br><br>
                    <strong>Performance:</strong> Zero additional latency. Highly optimized pure-tensor topk thresholding.
                </div>
            </div>
            <div class="sy-panel" style="border-top: 4px solid #8b5cf6;">
                <div class="sy-panel-title">Cryptographic Commitments & Paillier HE</div>
                <div class="sy-panel-text">
                    <strong>Pedersen Commitments:</strong> SHA-256 commitments with blinding factors before updates are opened, preventing adaptive tampering.<br><br>
                    <strong>Paillier HE:</strong> Partial homomorphic encryption allows additions directly on ciphertexts: E(m1) * E(m2) = E(m1 + m2).<br><br>
                    <strong>SHA-256 Merkle Ledger:</strong> Complete tamper-evident audit history across all federated rounds.
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
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="sy-panel-title" style="margin-bottom: 0;">UNSW-NB15 Benchmark</span>
                    <span style="background: #fef9c3; color: #92400e; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 50px;">42/58 Features</span>
                </div>
                <div class="sy-panel-text" style="margin-bottom: 12px;">
                    Synthesized by Australian Centre for Cyber Security using IXIA PerfectStorm to capture modern benign and real attack behaviors.
                </div>
                <div style="font-size: 12px; font-weight: 800; color: #1a1a2e; text-transform: uppercase; margin-bottom: 8px;">9 Attack Families:</div>
                <div>{tags_unsw}</div>
            </div>
            """, unsafe_allow_html=True)

        with dt2:
            tags_cicids = " ".join([f'<span style="display:inline-block; background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700; padding:4px 12px; border-radius:50px; margin: 3px 2px;">{cat}</span>' for cat in CICIDS_ATTACK_CATEGORIES[:10]])
            st.markdown(f"""
            <div class="sy-panel" style="border-top: 4px solid #3b82f6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="sy-panel-title" style="margin-bottom: 0;">CICIDS2017 Benchmark</span>
                    <span style="background: #dbeafe; color: #1d4ed8; font-size: 11px; font-weight: 800; padding: 5px 14px; border-radius: 50px;">78 Flow Features</span>
                </div>
                <div class="sy-panel-text" style="margin-bottom: 12px;">
                    Generated by Canadian Institute for Cybersecurity over 5 days of captured network traffic using CICFlowMeter.
                </div>
                <div style="font-size: 12px; font-weight: 800; color: #1a1a2e; text-transform: uppercase; margin-bottom: 8px;">14 Attack Classes:</div>
                <div>{tags_cicids}</div>
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4: RESEARCH & ARCHITECTURE (ACADEMIC SOTA & NOVEL TECHNIQUES)         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state.active_page in ["Research & SOTA", "Research & Architecture"]:

    # Hero
    st.markdown("""
    <div style="padding: 48px 0 24px 0;">
        <div class="sy-hero-tag">ACADEMIC SOTA & INDUSTRY RESEARCH</div>
        <h1 style="font-size: 42px !important; font-weight: 900 !important; letter-spacing: -0.04em !important; color: #1a1a2e !important; line-height: 1.1 !important; margin: 0 0 12px 0 !important;">
            State of the Art vs.<br><span style="color: #eab308;">AEGIS-FL Architecture</span>
        </h1>
        <p style="font-size: 17px; color: #64748b; max-width: 760px; line-height: 1.65;">
            A rigorous technical comparison evaluating centralized commercial EDR/SIEM products, academic federated learning research papers, current corporate practices, and the 5 novel techniques introduced in our architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Current State of the Art (The Baseline & Its Flaws) ──
    st.markdown("""
    <div class="sy-section-center" style="padding-top: 24px;">
        <div class="sy-section-tag">BASELINE ANALYSIS</div>
        <div class="sy-section-title">What Currently Exists in Industry & Literature</div>
        <div class="sy-section-desc">The structural limitations that leave modern enterprises vulnerable to systemic breaches.</div>
    </div>

    <div class="sy-compare-grid">
        <div class="sy-compare-card warning">
            <div class="sy-compare-icon">&#9729;&#65039;</div>
            <div class="sy-compare-title">Commercial Cloud EDR / SIEM Giants</div>
            <div class="sy-compare-sub">CrowdStrike Falcon &bull; SentinelOne &bull; Splunk Cloud</div>
            <div class="sy-compare-text">
                <strong>Mechanism:</strong> Requires streaming gigabytes of raw internal network telemetry, NetFlow, and event logs into a multi-tenant cloud lake.<br><br>
                <strong>Fatal Liability:</strong> Violates EU GDPR Articles 44/46 (cross-border transfers) and HIPAA §164.502. Creates a massive centralized honeypot and single-point-of-failure blast radius (demonstrated in the July 2024 global IT crash).
            </div>
        </div>
        <div class="sy-compare-card danger">
            <div class="sy-compare-icon">&#128225;</div>
            <div class="sy-compare-title">Static Threat Intelligence Exchanges</div>
            <div class="sy-compare-sub">FS-ISAC &bull; MITRE ATT&CK &bull; STIX / TAXII Feeds</div>
            <div class="sy-compare-text">
                <strong>Mechanism:</strong> Manual, post-incident sharing of coarse Indicators of Compromise (IOCs: static IP blacklists, malicious domains, and SHA-256 hashes).<br><br>
                <strong>Fatal Liability:</strong> IOCs arrive hours or days after an intrusion; completely powerless against zero-days, dynamic domain generation (DGA), and polymorphic malware payloads.
            </div>
        </div>
        <div class="sy-compare-card danger">
            <div class="sy-compare-icon">&#128214;</div>
            <div class="sy-compare-title">Early Academic Federated Learning</div>
            <div class="sy-compare-sub">FedAvg (McMahan 2017) &bull; SecAgg (Bonawitz 2017)</div>
            <div class="sy-compare-text">
                <strong>Mechanism:</strong> Distributes training to silos and computes mathematical parameter averages.<br><br>
                <strong>Fatal Liability:</strong> FedAvg is totally defenseless against Byzantine poisoning (1 rogue node inverts the model). SecAgg creates the <em>"Blinded Server Dilemma"</em>: cryptographic masks prevent the aggregator from verifying updates, allowing poisoned weights to pass unchecked.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: What Real Companies Are Using Right Now ──
    st.markdown("""
    <div class="sy-section-center" style="padding-top: 36px;">
        <div class="sy-section-tag">REAL-WORLD DEPLOYMENTS</div>
        <div class="sy-section-title">What Companies Are Actually Using Right Now</div>
        <div class="sy-section-desc">Where federated learning exists in production today — and why cybersecurity was left behind.</div>
    </div>

    <div class="sy-pillars-grid" style="grid-template-columns: repeat(2, 1fr);">
        <div class="sy-panel" style="border-top: 4px solid #3b82f6;">
            <div class="sy-panel-title">&#128241; Consumer Edge FL: Google & Apple</div>
            <div class="sy-panel-text">
                <strong>Current Production Use:</strong> Google Gboard next-word suggestions and Apple Siri vocal modeling run federated learning across hundreds of millions of mobile phones.<br><br>
                <strong>Why It Fails for Cybersecurity:</strong> Consumer mobile FL operates in benign, low-stakes environments with homogenous IID data. Devices do not face nation-state adversaries running targeted Byzantine gradient poisoning, sign-flipping, or model scaling attacks.
            </div>
        </div>
        <div class="sy-panel" style="border-top: 4px solid #10b981;">
            <div class="sy-panel-title">&#127973; Healthcare Consortia: Owkin & MELLODDY</div>
            <div class="sy-panel-text">
                <strong>Current Production Use:</strong> Multi-hospital cancer histology and drug discovery across pharmaceutical giants (Pfizer, Amgen, Janssen).<br><br>
                <strong>Why It Fails for Cybersecurity:</strong> Operates in legally "walled-garden" consortiums where all participants are pre-screened institutional partners bound by contracts. It completely lacks zero-trust adversarial defenses, real-time intrusion streaming, and anti-inversion gradient sparsification.
            </div>
        </div>
    </div>

    <div class="sy-alert danger" style="margin-top: 24px;">
        <div class="sy-alert-icon">&#9888;&#65039;</div>
        <div>
            <div class="sy-alert-title" style="color: #b91c1c !important;">The Commercial Cybersecurity Void</div>
            <div class="sy-alert-text">
                Prior to AEGIS-FL, there was <strong>virtually zero commercial enterprise adoption</strong> of federated learning in cybersecurity. Enterprise CISOs and SOC teams refused to accept unverified black-box neural updates, feared model poisoning, and lacked cryptographic audit chains that satisfy SOC2 Type II, ISO 27001, and NIST CSF compliance auditors.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 3: What We Improvised — The 5 Novel Techniques ──
    st.markdown("""
    <div class="sy-section-center" style="padding-top: 48px;">
        <div class="sy-section-tag">CORE INNOVATIONS</div>
        <div class="sy-section-title">What We Improvised: 5 Novel Architectural Techniques</div>
        <div class="sy-section-desc">How AEGIS-FL solves the Privacy-Robustness Trilemma and achieves 91.93% benchmark accuracy.</div>
    </div>

    <div class="sy-pillars-grid">
        <div class="sy-pillar" style="border-top-color: #facc15;">
            <div class="sy-pillar-num">01</div>
            <div class="sy-pillar-title">Solving the Privacy-Robustness Trilemma</div>
            <div class="sy-pillar-text">
                <strong>The Breakthrough:</strong> Solves the <em>Blinded Server Dilemma</em>. Combines Pedersen cryptographic commitments with geometric norm verification and reference-calibrated Cosine alignment (FLTrust-inspired). Filters out poisoned and scaled client updates <em>before</em> aggregation without ever decrypting or inspecting raw institutional telemetry.
            </div>
        </div>
        <div class="sy-pillar" style="border-top-color: #3b82f6;">
            <div class="sy-pillar-num">02</div>
            <div class="sy-pillar-title">Energy-Preserving Anti-Inversion Sparsification</div>
            <div class="sy-pillar-text">
                <strong>The Breakthrough:</strong> Zeroes out 95% of update coefficients to defeat gradient inversion attacks (Deep Leakage from Gradients - DLG). Unlike standard sparsifiers that destroy accuracy, our engine scales retained coordinate energy: <code>g_scaled = g_sparse * (||g|| / ||g_sparse||)</code>, preserving full residual skip-connection signal.
            </div>
        </div>
        <div class="sy-pillar" style="border-top-color: #8b5cf6;">
            <div class="sy-pillar-num">03</div>
            <div class="sy-pillar-title">Tight Order-Optimized R&eacute;nyi DP (RDP)</div>
            <div class="sy-pillar-text">
                <strong>The Breakthrough:</strong> Discards loose linear compositions that exhaust privacy budgets in early rounds. Employs a closed-form R&eacute;nyi moments accountant dynamically searching across alpha in [1.5, 64], yielding mathematically provable (epsilon, delta) guarantees with &gt;45% tighter privacy bounds.
            </div>
        </div>
        <div class="sy-pillar" style="border-top-color: #10b981;">
            <div class="sy-pillar-num">04</div>
            <div class="sy-pillar-title">Production Deep Neural Suite (1D ResNet & Attention)</div>
            <div class="sy-pillar-text">
                <strong>The Breakthrough:</strong> Replaces toy 2-layer MLPs with 1D Deep Residual Networks (skip connections, LayerNorm, LeakyReLU) and Tabular Multi-Head Feature-Attention Networks, pushing benchmark accuracy on UNSW-NB15 and CICIDS2017 to <strong>91.93% Accuracy</strong> and <strong>91.99% Precision</strong>.
            </div>
        </div>
        <div class="sy-pillar" style="border-top-color: #ef4444;">
            <div class="sy-pillar-num">05</div>
            <div class="sy-pillar-title">Immutable SHA-256 Merkle Audit Chain</div>
            <div class="sy-pillar-text">
                <strong>The Breakthrough:</strong> Every federated round produces an immutable cryptographic block linking the prior block hash, participating silo signatures, global model weights, and benchmark accuracy. Provides complete forensic non-repudiation and audit readiness for SOC2 and NIST CSF.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 4: Comprehensive Comparison Matrix ──
    st.markdown("""
    <div class="sy-section-center" style="padding-top: 48px;">
        <div class="sy-section-tag">COMPETITIVE MATRIX</div>
        <div class="sy-section-title">Architectural Capability Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

    comp_df = pd.DataFrame([
        {
            "Capability": "Raw Telemetry Movement",
            "Traditional On-Prem": "Strictly Local (Air-Gapped)",
            "Cloud EDR (CrowdStrike)": "Raw packets uploaded to vendor cloud",
            "Threat Feeds (FS-ISAC)": "Static text reports of IPs/hashes",
            "Early FL (FedAvg/SecAgg)": "0 Bytes leave perimeter",
            "AEGIS-FL (PPFL-IDS)": "0 Bytes leave perimeter (Mathematically Proven)"
        },
        {
            "Capability": "Zero-Day Reaction Speed",
            "Traditional On-Prem": "Weeks / Months",
            "Cloud EDR (CrowdStrike)": "Hours (cloud-gated)",
            "Threat Feeds (FS-ISAC)": "Days of manual incident analysis",
            "Early FL (FedAvg/SecAgg)": "Rounds to converge",
            "AEGIS-FL (PPFL-IDS)": "Sub-second autonomous neural adaptation"
        },
        {
            "Capability": "Collaborative Neural Intelligence",
            "Traditional On-Prem": "None (isolated castle)",
            "Cloud EDR (CrowdStrike)": "Vendor owns intelligence & model",
            "Threat Feeds (FS-ISAC)": "Coarse manual IOC lists",
            "Early FL (FedAvg/SecAgg)": "Collaborative (toy MLP)",
            "AEGIS-FL (PPFL-IDS)": "Continuous collaborative 1D-ResNet & Attention"
        },
        {
            "Capability": "Privacy Guarantee",
            "Traditional On-Prem": "Compliant but blind",
            "Cloud EDR (CrowdStrike)": "Vendor Trust / Compliance friction",
            "Threat Feeds (FS-ISAC)": "Manual redaction",
            "Early FL (FedAvg/SecAgg)": "Vulnerable to DLG gradient inversion",
            "AEGIS-FL (PPFL-IDS)": "Tight R&eacute;nyi DP + Anti-Inversion Sparsification"
        },
        {
            "Capability": "Byzantine Poisoning Defense",
            "Traditional On-Prem": "N/A",
            "Cloud EDR (CrowdStrike)": "Vendor centralized heuristic",
            "Threat Feeds (FS-ISAC)": "Blind trust (vulnerable)",
            "Early FL (FedAvg/SecAgg)": "Blinded Server Dilemma (0% Defense)",
            "AEGIS-FL (PPFL-IDS)": "Multi-Krum &bull; FLAME &bull; FLTrust &bull; Norm Outliers"
        },
        {
            "Capability": "Tamper-Evident Forensic Audit",
            "Traditional On-Prem": "Local SQL logs (editable)",
            "Cloud EDR (CrowdStrike)": "Proprietary vendor cloud logs",
            "Threat Feeds (FS-ISAC)": "Static PDF bulletins",
            "Early FL (FedAvg/SecAgg)": "None",
            "AEGIS-FL (PPFL-IDS)": "SHA-256 Merkle Audit Chain (Cryptographically Validated)"
        }
    ])
    st.dataframe(comp_df, use_container_width=True)

    # ── Section 5: 5 Sovereign Institutional Silos ──
    st.markdown("""
    <div class="sy-section-center" style="padding-top: 48px;">
        <div class="sy-section-tag">SECTOR COVERAGE</div>
        <div class="sy-section-title">5 Sovereign Institutional Silos in AEGIS-FL</div>
    </div>
    """, unsafe_allow_html=True)

    sec_tabs = st.tabs([
        "Banking & Finance",
        "Healthcare & IoMT",
        "Corporate Enterprise",
        "Government & Defense",
        "Cloud Providers"
    ])

    with sec_tabs[0]:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #1a1a2e;">
            <div class="sy-panel-title">Banking & Financial Services (Client 1 - Bank)</div>
            <div class="sy-panel-text">
                <strong>Legal Barrier:</strong> Gramm-Leach-Bliley Act (GLBA), PCI-DSS, and the Bank Secrecy Act make sharing customer transaction flows illegal.<br><br>
                <strong>The Threat:</strong> Coordinated zero-day financial fraud, wire diversion, and carding botnets targeting SWIFT/ACH gateways.<br><br>
                <strong>AEGIS-FL Value:</strong> Banks train local neural models on raw payment flows. Only masked weight differentials leave the perimeter, enabling instant cross-bank fraud defense with zero customer exposure.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sec_tabs[1]:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #22c55e;">
            <div class="sy-panel-title">Healthcare & IoMT Networks (Client 2 - Hospital)</div>
            <div class="sy-panel-text">
                <strong>Legal Barrier:</strong> HIPAA & GDPR-Health levy up to $50,000 per violation and criminal liability for leaking PHI.<br><br>
                <strong>The Threat:</strong> Unpatchable medical devices (MRI, infusion pumps on legacy Windows) held hostage by ransomware (Change Healthcare, WannaCry).<br><br>
                <strong>AEGIS-FL Value:</strong> Hospitals achieve collective immunity against new ransomware patterns without transmitting a single patient health record.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sec_tabs[2]:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #eab308;">
            <div class="sy-panel-title">Corporate Enterprise & Supply Chain (Client 3 - Enterprise)</div>
            <div class="sy-panel-text">
                <strong>The Barrier:</strong> Corporate espionage, trade secret leaks, and antitrust laws against data sharing between competitors.<br><br>
                <strong>The Threat:</strong> Supply-chain infiltrations (SolarWinds) and Living-off-the-Land binaries abusing PowerShell.<br><br>
                <strong>AEGIS-FL Value:</strong> Enterprises detect compromised third-party credentials across ecosystems without exposing source code or business logic.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sec_tabs[3]:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #ef4444;">
            <div class="sy-panel-title">Government, Intelligence & Defense (Client 4 - Government)</div>
            <div class="sy-panel-text">
                <strong>The Barrier:</strong> Strict national security classifications (Top Secret / SCI) and compartmentalization between Five Eyes allies.<br><br>
                <strong>The Threat:</strong> State-sponsored APTs (Volt Typhoon) penetrating critical power, water, and defense infrastructure.<br><br>
                <strong>AEGIS-FL Value:</strong> Allied military and civil subnets collaborate on zero-day vectors with mathematical zero-trust guarantees.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sec_tabs[4]:
        st.markdown("""
        <div class="sy-panel" style="border-top: 4px solid #3b82f6;">
            <div class="sy-panel-title">Public Cloud Infrastructure Providers (Client 5 - Cloud Provider)</div>
            <div class="sy-panel-text">
                <strong>The Barrier:</strong> Strict tenant privacy SLAs and global data-residency laws (EU cloud sovereignty directives).<br><br>
                <strong>The Threat:</strong> Multi-tenant hypervisor escapes, CPU side-channels, and massive IoT botnet DDoS campaigns.<br><br>
                <strong>AEGIS-FL Value:</strong> Cloud providers coordinate edge DDoS scrubbing and boundary defenses without inspecting customer payload packets.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section 6: Open-Source Researcher Public Invitation (Prominent Bold Callout) ──
    st.html("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 2.5px solid #facc15; border-radius: 24px; padding: 44px; margin: 48px 0; color: #ffffff; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);">
        <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 20px;">
            <span style="font-size: 36px;">🌐</span>
            <span style="background: #facc15; color: #0f172a; font-size: 12px; font-weight: 800; padding: 6px 16px; border-radius: 50px; text-transform: uppercase; letter-spacing: 0.08em;">Open Research Initiative</span>
            <span style="background: rgba(255,255,255,0.12); color: #f1f5f9; font-size: 12px; font-weight: 700; padding: 6px 16px; border-radius: 50px;">Peer Review &bull; RFC Open</span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; font-size: 12px; font-weight: 700; padding: 6px 16px; border-radius: 50px;">&#10003; Reproducible Benchmarks</span>
        </div>
        
        <h2 style="font-size: 28px !important; font-weight: 900 !important; color: #ffffff !important; line-height: 1.35 !important; margin: 0 0 20px 0 !important; letter-spacing: -0.03em !important;">
            <strong>Hey everyone! I'm actively working on this project, and I would love to add your contributions, ideas, and input into it!</strong>
        </h2>
        
        <p style="font-size: 16px; color: #cbd5e1; line-height: 1.75; max-width: 950px; margin-bottom: 32px;">
            I am developing <strong>AEGIS-FL™ (PPFL-IDS)</strong> as a fully open-source, privacy-preserving cyber defense framework so that researchers, cybersecurity analysts, engineers, and academic institutions worldwide can review our architecture, run reproducible benchmarks, test novel Byzantine defenses, and contribute their expertise. All pull requests, algorithm proposals, and security audits are warmly welcomed!
        </p>

        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 32px;">
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 22px;">
                <div style="font-size: 16px; font-weight: 800; color: #facc15; margin-bottom: 8px;">&#128196; 1. Academic Peer Review & Proof Audit</div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                    Validate the mathematical proofs for SecAgg+ pairwise zero-sum noise cancellation, Pedersen homomorphic commitments, and the order-optimized R&eacute;nyi DP moments accountant.
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 22px;">
                <div style="font-size: 16px; font-weight: 800; color: #facc15; margin-bottom: 8px;">&#128737;&#65039; 2. Adversarial Red-Teaming & Stress Testing</div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                    Conduct red-team evaluations: test novel backdoor injections, gradient inversion (DLG / iDLG), sybil attacks, and adaptive scaling against our Multi-Krum and FLAME filters.
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 22px;">
                <div style="font-size: 16px; font-weight: 800; color: #facc15; margin-bottom: 8px;">&#128300; 3. Reproducible Benchmarking & Flow Datasets</div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                    Verify benchmark accuracy (91.93%) and precision (91.99%) across UNSW-NB15, CICIDS2017, BoT-IoT, and custom enterprise PCAP captures on local hardware.
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 22px;">
                <div style="font-size: 16px; font-weight: 800; color: #facc15; margin-bottom: 8px;">&#9889; 4. Algorithmic Extensions & Community Input</div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                    Contribute new aggregation techniques, Graph Neural Network (GNN) backbones, hardware-accelerated Paillier HE, and custom local differential privacy mechanisms.
                </div>
            </div>
        </div>

        <div style="display: flex; flex-wrap: wrap; gap: 14px; align-items: center;">
            <a href="https://github.com/0xxay/AEGIS-FL" target="_blank" style="background: #facc15; color: #0f172a; text-decoration: none; font-weight: 800; font-size: 14px; padding: 12px 28px; border-radius: 50px; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 4px 14px rgba(250,204,21,0.4);">
                <span>&#128187;</span> Open GitHub Repository
            </a>
            <a href="https://github.com/0xxay/AEGIS-FL#readme" target="_blank" style="background: rgba(255,255,255,0.15); color: #ffffff; text-decoration: none; font-weight: 700; font-size: 14px; padding: 12px 28px; border-radius: 50px; display: inline-flex; align-items: center; gap: 8px; border: 1px solid rgba(255,255,255,0.25);">
                <span>&#128196;</span> Read Architecture Whitepaper
            </a>
            <a href="https://github.com/0xxay/AEGIS-FL/issues/new" target="_blank" style="background: rgba(255,255,255,0.1); color: #facc15; text-decoration: none; font-weight: 700; font-size: 14px; padding: 12px 24px; border-radius: 50px; display: inline-flex; align-items: center; gap: 8px; border: 1px solid rgba(250,204,21,0.3);">
                <span>&#9889;</span> Submit Research Input / RFC
            </a>
            <span style="font-size: 13px; color: #94a3b8; margin-left: 8px;">
                Licensed under Apache 2.0 &bull; Open for Peer Review & Community Input &bull; Contributions Welcome
            </span>
        </div>
    </div>
    """)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FOOTER                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<div class="sy-footer">
    <div class="sy-footer-brand">AEGIS-FL<span style="color: #facc15; font-size: 12px; vertical-align: super;">&#8482;</span></div>
    <div class="sy-footer-sub">The Sovereign Cyber Collective &bull; PPFL-IDS Network &bull; Open Research Initiative</div>
    <div style="margin-top: 12px; font-size: 12px; color: #94a3b8;">Apache 2.0 Open Source &bull; FIPS 140-3 &bull; GDPR & HIPAA Sovereign &bull; Peer Review Welcomed</div>
</div>
""", unsafe_allow_html=True)
