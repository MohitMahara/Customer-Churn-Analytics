import sqlite3
from datetime import datetime
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="CustomerChurn Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_INPUT_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "Number of Referrals",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "StreamingTV",
    "StreamingMovies",
    "Age",
    "Married",
]

DB_PATH = "churn_predictions.db"
RISK_COLORS = {
    "Low": "#34d399",
    "Medium": "#fbbf24",
    "High": "#fb7185",
}


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0b1020;
            --surface: #11182b;
            --surface-2: #151f36;
            --line: rgba(226, 232, 240, 0.12);
            --text: #edf2ff;
            --muted: #9aa7bd;
            --accent: #38bdf8;
            --accent-2: #a7f3d0;
            --accent-glow: rgba(56, 189, 248, 0.32);
            --danger: #fb7185;
            --warning: #fbbf24;
            --success: #34d399;
            --space-xs: 0.5rem;
            --space-sm: 0.75rem;
            --space-md: 1rem;
            --space-lg: 1.5rem;
            --space-xl: 2rem;
            --space-2xl: 2.75rem;
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
            background: var(--bg);
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.13), transparent 30rem),
                linear-gradient(180deg, #0b1020 0%, #0d1323 48%, #090d18 100%);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 3rem;
        }

        header[data-testid="stHeader"] * {
            visibility: visible;
        }

        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            visibility: hidden;
        }

        button[data-testid="collapsedControl"],
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }

        .block-container {
            max-width: 1180px;
            padding: 2.25rem 2rem 4.5rem;
        }

        section[data-testid="stSidebar"] {
            background: rgba(9, 13, 24, 0.96);
            border-right: 1px solid var(--line);
            z-index: 99999;
        }

        section[data-testid="stSidebar"] .block-container {
            padding: 1.35rem 1rem;
        }

        .brand {
            padding: 0.25rem 0.35rem 1rem;
        }

        .brand-name {
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .page-intro {
            max-width: 760px;
            margin: 0 0 var(--space-lg);
            padding: 0;
        }

        .hero {
            min-height: 430px;
            display: grid;
            align-items: center;
            padding: 2.8rem 0 2rem;
        }

        .eyebrow {
            color: var(--accent-2);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero h1, .page-title {
            color: var(--text);
            font-size: clamp(2.5rem, 6vw, 5.15rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 0.98;
            margin: 0 0 1.35rem;
        }

        .page-title {
            font-size: clamp(2.25rem, 4.6vw, 4rem);
        }

        .hero-copy, .page-copy {
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 680px;
            margin-bottom: 2rem;
        }

        .page-intro .page-copy {
            margin-bottom: 0;
        }

        .section-heading {
            margin: var(--space-xl) 0 var(--space-md);
            padding-top: 0.1rem;
        }

        .section-heading.first {
            margin-top: var(--space-sm);
        }

        .section-heading h2 {
            color: var(--text);
            font-size: 1.32rem;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0 0 0.35rem;
        }

        .section-heading p {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.65;
            margin: 0;
            max-width: 740px;
        }

        .section {
            padding: 2.1rem 0;
        }

        .section h2 {
            color: var(--text);
            font-size: 1.65rem;
            font-weight: 750;
            letter-spacing: 0;
            margin: 0 0 0.7rem;
        }

        .section p {
            color: var(--muted);
            line-height: 1.75;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: var(--space-md);
            margin-top: var(--space-lg);
        }

        .feature-card, .panel, .result-card, .metric-card {
            background: rgba(17, 24, 43, 0.78);
            border: 1px solid var(--line);
            border-radius: 1rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
        }

        .feature-card {
            padding: 1.35rem;
            transition: transform 160ms ease, border-color 160ms ease;
        }

        .feature-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.32);
        }

        .feature-card h3 {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin: 0 0 0.45rem;
        }

        .feature-card p {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.65;
            margin: 0;
        }

        .panel {
            padding: var(--space-lg);
            margin-top: var(--space-md);
        }

        .panel-title {
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 800;
            margin: 0 0 0.45rem;
        }

        .panel-caption {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.7;
            margin: 0 0 1.35rem;
        }

        .metric-grid {
            display: grid;
            gap: var(--space-md);
            margin: var(--space-md) 0 var(--space-xl);
        }

        .metric-grid.three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .metric-grid.four {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }

        .result-card {
            padding: 1.35rem 1.4rem;
            height: 100%;
            min-height: 7rem;
            position: relative;
            overflow: hidden;
        }

        .result-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.75), transparent);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .result-value {
            color: var(--text);
            font-size: clamp(1.35rem, 2vw, 1.85rem);
            font-weight: 800;
            line-height: 1.1;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: var(--space-md);
        }

        .result-main {
            padding: var(--space-xs) 0 0;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.48rem 0.8rem;
            font-size: 0.86rem;
            font-weight: 800;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.045);
            margin-bottom: 1rem;
        }

        .risk-pill::before {
            content: "";
            width: 0.52rem;
            height: 0.52rem;
            border-radius: 999px;
            background: currentColor;
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.06);
        }

        .status-note {
            margin-top: var(--space-md);
            color: var(--accent-2);
            font-size: 0.88rem;
            font-weight: 700;
        }

        .chart-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin: 0 0 0.25rem;
        }

        .chart-caption {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.55;
            margin: 0 0 var(--space-sm);
        }

        .risk-high { color: var(--danger); }
        .risk-medium { color: var(--warning); }
        .risk-low { color: var(--success); }

        .summary-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: var(--space-md);
            margin-top: 0;
        }

        .summary-item {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid var(--line);
            border-radius: 0.85rem;
            padding: 1rem;
        }

        .summary-item span {
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        .summary-item strong {
            color: var(--text);
            font-size: 0.98rem;
        }

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            width: 100%;
            border: 0;
            border-radius: 999px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.38), transparent 28%),
                linear-gradient(135deg, #67e8f9 0%, #38bdf8 42%, #22d3ee 100%);
            color: #06111f;
            font-weight: 800;
            min-height: 3.15rem;
            padding: 0.86rem 1.35rem;
            box-shadow: 0 18px 44px rgba(34, 211, 238, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.35);
            transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            filter: brightness(1.06);
            box-shadow: 0 22px 56px rgba(34, 211, 238, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.4);
            color: #06111f;
        }

        .stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active {
            transform: translateY(0);
            color: #06111f;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            justify-content: flex-start;
            min-height: 3.05rem;
            border: 1px solid rgba(226, 232, 240, 0.1);
            border-radius: 0.95rem;
            background: rgba(255, 255, 255, 0.035);
            color: var(--muted);
            padding: 0.72rem 0.95rem;
            box-shadow: none;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(56, 189, 248, 0.08);
            border-color: rgba(56, 189, 248, 0.28);
            color: var(--text);
            transform: translateY(-1px);
            box-shadow: none;
        }

        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(167, 243, 208, 0.08));
            border-color: rgba(56, 189, 248, 0.42);
            color: var(--text);
            box-shadow: 0 16px 36px rgba(56, 189, 248, 0.12);
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 0.8rem;
            overflow: hidden;
            margin: var(--space-sm) 0 var(--space-md);
        }

        div[data-testid="stDataFrame"] {
            background: rgba(7, 11, 21, 0.42);
            box-shadow: none;
        }

        .analytics-table-wrap {
            width: 100%;
            max-height: 520px;
            overflow: auto;
            margin-top: var(--space-sm);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 0.85rem;
            background: rgba(7, 11, 21, 0.42);
        }

        .records-table {
            width: 100%;
            min-width: 900px;
            border-collapse: collapse;
            color: #d7e2f2;
            font-size: 0.86rem;
        }

        .records-table thead th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #101729;
            color: var(--muted);
            padding: 0.78rem 0.85rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.14);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-align: left;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .records-table tbody td {
            padding: 0.78rem 0.85rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.08);
            vertical-align: middle;
            white-space: nowrap;
        }

        .records-table tbody tr:nth-child(even) td {
            background: rgba(255, 255, 255, 0.018);
        }

        .records-table tbody tr:hover td {
            background: rgba(56, 189, 248, 0.055);
        }

        .table-risk {
            display: inline-flex;
            align-items: center;
            min-width: 4.7rem;
            justify-content: center;
            border-radius: 999px;
            padding: 0.32rem 0.62rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.04);
            font-weight: 800;
        }

        .table-risk.risk-high {
            background: rgba(251, 113, 133, 0.1);
            border-color: rgba(251, 113, 133, 0.24);
        }

        .table-risk.risk-medium {
            background: rgba(251, 191, 36, 0.1);
            border-color: rgba(251, 191, 36, 0.24);
        }

        .table-risk.risk-low {
            background: rgba(52, 211, 153, 0.1);
            border-color: rgba(52, 211, 153, 0.24);
        }

        .probability-cell {
            min-width: 8.5rem;
        }

        .probability-cell span {
            display: block;
            color: var(--text);
            font-weight: 800;
            margin-bottom: 0.32rem;
        }

        .probability-track {
            height: 0.42rem;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(226, 232, 240, 0.1);
        }

        .probability-fill {
            height: 100%;
            border-radius: inherit;
            background: var(--accent);
        }

        .probability-fill.risk-high { background: var(--danger); }
        .probability-fill.risk-medium { background: var(--warning); }
        .probability-fill.risk-low { background: var(--success); }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(17, 24, 43, 0.78);
            border: 1px solid rgba(226, 232, 240, 0.12);
            border-radius: 1.05rem;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.22);
            padding: var(--space-lg) !important;
            margin: var(--space-md) 0 var(--space-lg) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            gap: var(--space-md) !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: var(--space-md);
        }

        div[data-testid="stHorizontalBlock"] {
            gap: var(--space-lg);
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
            margin-top: var(--space-sm);
        }

        div[data-testid="stForm"] h3 {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin: 0.3rem 0 var(--space-sm);
        }

        div[data-testid="stFormSubmitButton"] {
            margin-top: var(--space-lg);
            margin-bottom: var(--space-sm);
        }

        div[data-testid="stNumberInput"],
        div[data-testid="stSelectbox"],
        div[data-testid="stFileUploader"] {
            margin-bottom: var(--space-sm);
        }

        label[data-testid="stWidgetLabel"] p {
            color: #c8d3e4;
            font-size: 0.84rem;
            font-weight: 700;
        }

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: 0.68rem !important;
            border-color: rgba(226, 232, 240, 0.16) !important;
            background-color: rgba(255, 255, 255, 0.04) !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(56, 189, 248, 0.55) !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.12) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            align-items: center;
            gap: var(--space-sm);
            margin: 0 0 var(--space-md) !important;
            padding: 0;
        }

        .stTabs [data-baseweb="tab"] {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 999px;
            border: 1px solid var(--line);
            color: #d7e2f2 !important;
            min-height: 2.55rem;
            margin: 0;
            padding: 0 1.05rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.28);
        }

        .stTabs [data-baseweb="tab"] p {
            color: #d7e2f2 !important;
            font-size: 0.86rem;
            font-weight: 800;
            line-height: 1;
            margin: 0;
        }

        .stTabs [aria-selected="true"] p {
            color: var(--text) !important;
        }

        .stTabs [data-baseweb="tab-border"],
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 0;
            padding-bottom: var(--space-xl);
        }

        .stTabs div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: 0 !important;
        }

        div[data-testid="stPlotlyChart"] {
            border-radius: 0.85rem;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.012);
        }

        h3 {
            color: var(--text);
            font-weight: 800;
            margin-top: var(--space-md);
            margin-bottom: var(--space-sm);
        }

        .soft-divider {
            height: 1px;
            background: rgba(226, 232, 240, 0.1);
            margin: var(--space-xl) 0;
        }

        @media (max-width: 760px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            .feature-grid, .summary-list, .result-grid, .metric-grid.three, .metric-grid.four { grid-template-columns: 1fr; }
            .hero { min-height: auto; padding-top: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    return {
        "kmeans": joblib.load("kmeans.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "model": load_model("churn_ann_model.h5"),
        "columns": joblib.load("columns.pkl"),
    }


def preprocess_customer_data(df, model_columns):
    clean_df = df.copy()
    processed = pd.DataFrame(0, index=clean_df.index, columns=model_columns)

    numeric_columns = ["tenure", "MonthlyCharges", "Number of Referrals", "Age"]
    for column in numeric_columns:
        if column in clean_df.columns and column in processed.columns:
            processed[column] = pd.to_numeric(clean_df[column], errors="coerce").fillna(0)

    for idx, row in clean_df.iterrows():
        contract = row.get("Contract")
        internet = row.get("InternetService")
        payment = row.get("PaymentMethod")

        if contract == "One year" and "Contract_One year" in processed.columns:
            processed.loc[idx, "Contract_One year"] = 1
        elif contract == "Two year" and "Contract_Two year" in processed.columns:
            processed.loc[idx, "Contract_Two year"] = 1

        if internet == "Fiber optic" and "InternetService_Fiber optic" in processed.columns:
            processed.loc[idx, "InternetService_Fiber optic"] = 1
        elif internet == "No" and "InternetService_No" in processed.columns:
            processed.loc[idx, "InternetService_No"] = 1

        if payment == "Electronic check" and "PaymentMethod_Electronic check" in processed.columns:
            processed.loc[idx, "PaymentMethod_Electronic check"] = 1

        binary_mappings = {
            "OnlineSecurity": "OnlineSecurity_Yes",
            "StreamingTV": "StreamingTV_Yes",
            "StreamingMovies": "StreamingMovies_Yes",
            "Married": "Married_Yes",
        }
        for source, target in binary_mappings.items():
            if row.get(source) == "Yes" and target in processed.columns:
                processed.loc[idx, target] = 1

    return processed


def predict_churn(input_df, artifacts):
    processed = preprocess_customer_data(input_df, artifacts["columns"])
    scaled = artifacts["scaler"].transform(processed)
    probabilities = artifacts["model"].predict(scaled, verbose=0).flatten()
    clusters = artifacts["kmeans"].predict(scaled)
    return probabilities, clusters


def risk_level(probability):
    if probability > 0.7:
        return "High"
    if probability > 0.4:
        return "Medium"
    return "Low"


def risk_class(risk):
    return {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(risk, "")


def risk_color(risk):
    return RISK_COLORS.get(risk, "#38bdf8")


def revenue_at_risk(probability, monthly_charges):
    return float(probability) * float(monthly_charges)


def format_currency(value):
    return f"${value:,.2f}"


def compact_currency(value):
    value = float(value or 0)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def chart_layout(fig, height=300, show_legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=16, t=22, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c8d3e4", size=12),
        showlegend=show_legend,
        bargap=0.34,
        hoverlabel=dict(
            bgcolor="#11182b",
            bordercolor="rgba(226, 232, 240, 0.18)",
            font=dict(family="Inter", color="#edf2ff", size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#c8d3e4"),
        ),
        xaxis=dict(
            gridcolor="rgba(226, 232, 240, 0.07)",
            zeroline=False,
            linecolor="rgba(226, 232, 240, 0.14)",
            tickfont=dict(size=11, color="#9aa7bd"),
            title_font=dict(size=12, color="#c8d3e4"),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(226, 232, 240, 0.07)",
            zeroline=False,
            linecolor="rgba(226, 232, 240, 0.14)",
            tickfont=dict(size=11, color="#9aa7bd"),
            title_font=dict(size=12, color="#c8d3e4"),
            automargin=True,
        ),
    )
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(family="Inter", color="#edf2ff", size=12),
        cliponaxis=False,
    )
    return fig


def render_plotly_chart(fig, height=300, show_legend=False):
    st.plotly_chart(
        chart_layout(fig, height=height, show_legend=show_legend),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )


def probability_gauge(probability):
    risk = risk_level(probability)
    probability_percent = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={
                "suffix": "%",
                "font": {"family": "Inter", "size": 42, "color": "#edf2ff"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "rgba(226, 232, 240, 0.35)",
                    "tickfont": {"color": "#9aa7bd", "size": 10},
                },
                "bar": {"color": risk_color(risk), "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(52, 211, 153, 0.14)"},
                    {"range": [40, 70], "color": "rgba(251, 191, 36, 0.14)"},
                    {"range": [70, 100], "color": "rgba(251, 113, 133, 0.14)"},
                ],
                "threshold": {
                    "line": {"color": "#edf2ff", "width": 2},
                    "thickness": 0.72,
                    "value": probability_percent,
                },
            },
        )
    )
    return fig


def risk_distribution_chart(values):
    risk_order = ["Low", "Medium", "High"]
    counts = values.value_counts().reindex(risk_order).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(
                color=[RISK_COLORS[label] for label in counts.index],
                line=dict(color="rgba(255,255,255,0.12)", width=1),
            ),
            text=[f"{int(value):,}" for value in counts.values],
            textposition="outside",
            hovertemplate="%{x} risk<br>%{y:,} customers<extra></extra>",
        )
    )
    max_count = max(float(counts.max()), 1.0)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Customers", range=[0, max_count * 1.25])
    if max_count <= 8:
        fig.update_yaxes(dtick=1)
    return fig


def probability_histogram(probabilities):
    probability_percent = pd.to_numeric(probabilities, errors="coerce").dropna() * 100
    fig = go.Figure(
        go.Histogram(
            x=probability_percent,
            xbins=dict(start=0, end=100, size=10),
            marker=dict(color="rgba(56, 189, 248, 0.72)", line=dict(color="rgba(255,255,255,0.12)", width=1)),
            hovertemplate="Probability band: %{x:.1f}%<br>Customers: %{y}<extra></extra>",
        )
    )
    fig.add_vrect(x0=0, x1=40, fillcolor="rgba(52, 211, 153, 0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=40, x1=70, fillcolor="rgba(251, 191, 36, 0.08)", line_width=0, layer="below")
    fig.add_vrect(x0=70, x1=100, fillcolor="rgba(251, 113, 133, 0.08)", line_width=0, layer="below")
    fig.add_vline(x=40, line_dash="dash", line_color="#fbbf24", opacity=0.9)
    fig.add_vline(x=70, line_dash="dash", line_color="#fb7185", opacity=0.95)
    fig.update_xaxes(title_text="Churn probability", ticksuffix="%")
    fig.update_yaxes(title_text="Customers")
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(range=[0, 100])
    return fig


def cluster_risk_chart(records):
    cluster_df = (
        records.groupby("cluster", as_index=False)
        .agg(avg_risk=("churn_probability", "mean"), customers=("cluster", "count"))
        .sort_values("cluster")
    )
    colors = [risk_color(risk_level(value)) for value in cluster_df["avg_risk"]]
    fig = go.Figure(
        go.Bar(
            x=cluster_df["cluster"].astype(str),
            y=cluster_df["avg_risk"] * 100,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.12)", width=1)),
            text=[f"{value * 100:.1f}%" for value in cluster_df["avg_risk"]],
            textposition="outside",
            customdata=cluster_df["customers"],
            hovertemplate="Cluster %{x}<br>Avg risk: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>",
        )
    )
    max_risk = max(float((cluster_df["avg_risk"] * 100).max()), 1.0)
    fig.update_xaxes(title_text="Customer cluster")
    fig.update_yaxes(title_text="Average risk", ticksuffix="%", range=[0, min(100, max_risk * 1.28)])
    return fig


def revenue_by_risk_chart(records):
    risk_order = ["Low", "Medium", "High"]
    revenue = records.groupby("risk_level")["revenue_at_risk"].sum().reindex(risk_order).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=revenue.index,
            y=revenue.values,
            marker=dict(
                color=[RISK_COLORS[label] for label in revenue.index],
                line=dict(color="rgba(255,255,255,0.12)", width=1),
            ),
            text=[compact_currency(value) for value in revenue.values],
            textposition="outside",
            hovertemplate="%{x}<br>Revenue at risk: $%{y:,.2f}<extra></extra>",
        )
    )
    max_revenue = max(float(revenue.max()), 1.0)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(range=[0, max_revenue * 1.25])
    fig.update_yaxes(title_text="Revenue at risk", tickprefix="$")
    return fig


def html_text(value):
    if pd.isna(value):
        return "-"
    return escape(str(value))


def table_number(value, suffix="", decimals=0):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"{numeric:,.{decimals}f}{suffix}"


def table_currency(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "-"
    return f"${numeric:,.2f}"


def render_recent_records_table(records):
    table = records[
        [
            "created_at",
            "source",
            "tenure",
            "monthly_charges",
            "contract",
            "risk_level",
            "churn_probability",
            "cluster",
            "revenue_at_risk",
        ]
    ].head(50).copy()

    rows = []
    for _, row in table.iterrows():
        saved_at = pd.to_datetime(row.get("created_at"), errors="coerce")
        saved_text = "Unknown" if pd.isna(saved_at) else saved_at.strftime("%d %b %Y, %I:%M %p")
        risk = "Unknown" if pd.isna(row.get("risk_level")) else str(row.get("risk_level"))
        risk_css = risk_class(risk)
        probability = pd.to_numeric(row.get("churn_probability"), errors="coerce")
        probability_percent = 0.0 if pd.isna(probability) else float(np.clip(probability * 100, 0, 100))
        probability_text = f"{probability_percent:.1f}%"

        rows.append(
            "<tr>"
            f"<td>{escape(saved_text)}</td>"
            f"<td>{html_text(row.get('source'))}</td>"
            f"<td>{table_number(row.get('tenure'), suffix=' mo')}</td>"
            f"<td>{table_currency(row.get('monthly_charges'))}</td>"
            f"<td>{html_text(row.get('contract'))}</td>"
            f'<td><span class="table-risk {risk_css}">{escape(risk)}</span></td>'
            '<td><div class="probability-cell">'
            f"<span>{probability_text}</span>"
            '<div class="probability-track">'
            f'<div class="probability-fill {risk_css}" style="width: {probability_percent:.1f}%"></div>'
            "</div></div></td>"
            f"<td>{table_number(row.get('cluster'))}</td>"
            f"<td>{table_currency(row.get('revenue_at_risk'))}</td>"
            "</tr>"
        )

    table_html = (
        '<div class="analytics-table-wrap">'
        '<table class="records-table">'
        "<thead><tr>"
        "<th>Saved</th>"
        "<th>Source</th>"
        "<th>Tenure</th>"
        "<th>Monthly charges ($)</th>"
        "<th>Contract</th>"
        "<th>Risk</th>"
        "<th>Probability</th>"
        "<th>Cluster</th>"
        "<th>Revenue at risk ($)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def init_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                tenure REAL,
                monthly_charges REAL,
                referrals REAL,
                contract TEXT,
                internet_service TEXT,
                payment_method TEXT,
                online_security TEXT,
                streaming_tv TEXT,
                streaming_movies TEXT,
                age REAL,
                married TEXT,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                cluster INTEGER NOT NULL,
                revenue_at_risk REAL NOT NULL
            )
            """
        )


def save_prediction_records(result_df, source):
    rows = []
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _, row in result_df.iterrows():
        rows.append(
            (
                created_at,
                source,
                row.get("tenure"),
                row.get("MonthlyCharges"),
                row.get("Number of Referrals"),
                row.get("Contract"),
                row.get("InternetService"),
                row.get("PaymentMethod"),
                row.get("OnlineSecurity"),
                row.get("StreamingTV"),
                row.get("StreamingMovies"),
                row.get("Age"),
                row.get("Married"),
                float(row.get("Churn Probability", 0)),
                row.get("Risk Level", "Unknown"),
                int(row.get("Cluster", 0)),
                float(row.get("Revenue at Risk", 0)),
            )
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT INTO predictions (
                created_at, source, tenure, monthly_charges, referrals,
                contract, internet_service, payment_method, online_security,
                streaming_tv, streaming_movies, age, married, churn_probability,
                risk_level, cluster, revenue_at_risk
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_saved_predictions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn,
        )


def build_prediction_frame(raw_df, artifacts):
    probabilities, clusters = predict_churn(raw_df, artifacts)
    result_df = raw_df.copy()
    result_df["Churn Probability"] = probabilities
    result_df["Churn Probability %"] = (probabilities * 100).round(1)
    result_df["Risk Level"] = [risk_level(probability) for probability in probabilities]
    result_df["Cluster"] = clusters

    if "MonthlyCharges" in result_df.columns:
        monthly = pd.to_numeric(result_df["MonthlyCharges"], errors="coerce").fillna(0)
        result_df["Revenue at Risk"] = probabilities * monthly
    else:
        result_df["Revenue at Risk"] = 0.0

    return result_df


def render_sidebar():
    with st.sidebar:
        try:
            query_page = st.query_params.get("page", "Home")
        except AttributeError:
            query_params = st.experimental_get_query_params()
            query_page = query_params.get("page", ["Home"])
        if isinstance(query_page, list):
            query_page = query_page[0]
        valid_pages = {"Home", "Predict", "Analytics"}
        if "page" not in st.session_state:
            st.session_state.page = query_page if query_page in valid_pages else "Home"
        page = st.session_state.page if st.session_state.page in valid_pages else "Home"

        st.markdown(
            """
            <div class="brand">
                <div class="brand-name">CustomerChurn Analytics</div>
                <div class="brand-subtitle">Retention risk prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in ["Home", "Predict", "Analytics"]:
            button_type = "primary" if item == page else "secondary"
            if st.button(item, key=f"nav_{item.lower()}", type=button_type, use_container_width=True):
                st.session_state.page = item
                try:
                    st.query_params["page"] = item
                except AttributeError:
                    st.experimental_set_query_params(page=item)
                st.rerun()

        st.markdown("---")
        st.caption("Saved predictions are stored locally in SQLite for analytics.")
        return st.session_state.page


def render_page_intro(eyebrow, title, copy):
    st.markdown(
        '<div class="page-intro">'
        f'<div class="eyebrow">{escape(str(eyebrow))}</div>'
        f'<h1 class="page-title">{escape(str(title))}</h1>'
        f'<p class="page-copy">{escape(str(copy))}</p>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_section_heading(title, caption=None, first=False):
    first_class = " first" if first else ""
    caption_html = f"<p>{escape(str(caption))}</p>" if caption else ""
    st.markdown(
        f'<div class="section-heading{first_class}">'
        f"<h2>{escape(str(title))}</h2>"
        f"{caption_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics, columns=3):
    grid_class = "four" if columns == 4 else "three"
    cards = []
    for label, value, class_name in metrics:
        safe_class = " ".join(token for token in str(class_name).split() if token.replace("-", "").isalnum())
        cards.append(
            '<div class="result-card">'
            f'<div class="result-label">{escape(str(label))}</div>'
            f'<div class="result-value {safe_class}">{escape(str(value))}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="metric-grid {grid_class}">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_home():
    st.markdown(
        """
        <section class="hero">
            <div>
                <div class="eyebrow">Customer retention platform</div>
                <h1>CustomerChurn Analytics</h1>
                <p class="hero-copy">
                    Churn Studio turns customer attributes into clear churn probability,
                    segment context, and revenue exposure so retention teams can act with focus.
                </p>
            </div>
        </section>
        <section class="section">
            <h2>Built for quick decisions</h2>
            <p>
                Upload a customer CSV or score one customer manually. The app keeps the
                prediction workflow simple: clean inputs, readable outputs, and no dashboard clutter.
            </p>
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>Score individual customers</h3>
                    <p>Use a compact form to estimate churn probability, risk band, cluster, and monthly revenue at risk.</p>
                </div>
                <div class="feature-card">
                    <h3>Analyze CSV files</h3>
                    <p>Batch-score customer records and review a styled table that is ready for export.</p>
                </div>
                <div class="feature-card">
                    <h3>Keep the model intact</h3>
                    <p>The existing scaler, ANN model, KMeans model, and saved training columns remain the source of truth.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_single_prediction_result(probability, risk, cluster, revenue):
    gauge_col, detail_col = st.columns([1.05, 1.25], gap="large")
    with gauge_col:
        render_plotly_chart(probability_gauge(probability), height=285)

    with detail_col:
        st.markdown(
            f"""
            <div class="result-main">
                <div class="risk-pill {risk_class(risk)}">{risk} churn risk</div>
                <div class="result-grid">
                    <div class="result-card">
                        <div class="result-label">Churn Probability</div>
                        <div class="result-value">{probability:.1%}</div>
                    </div>
                    <div class="result-card">
                        <div class="result-label">Cluster</div>
                        <div class="result-value">{cluster}</div>
                    </div>
                    <div class="result-card">
                        <div class="result-label">Revenue at Risk</div>
                        <div class="result-value">{format_currency(revenue)}</div>
                    </div>
                    <div class="result-card">
                        <div class="result-label">Action Priority</div>
                        <div class="result-value {risk_class(risk)}">{risk}</div>
                    </div>
                </div>
                <div class="status-note">Saved to the local prediction database.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chart_header(title, caption):
    st.markdown(
        f"""
        <div class="chart-title">{title}</div>
        <div class="chart-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_summary(customer):
    st.markdown(
        f"""
        <div class="summary-list">
            <div class="summary-item"><span>Tenure</span><strong>{customer["tenure"]} months</strong></div>
            <div class="summary-item"><span>Monthly Charges</span><strong>{format_currency(customer["MonthlyCharges"])}</strong></div>
            <div class="summary-item"><span>Contract</span><strong>{customer["Contract"]}</strong></div>
            <div class="summary-item"><span>Internet Service</span><strong>{customer["InternetService"]}</strong></div>
            <div class="summary-item"><span>Payment Method</span><strong>{customer["PaymentMethod"]}</strong></div>
            <div class="summary-item"><span>Referrals</span><strong>{customer["Number of Referrals"]}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manual_input(artifacts):
    st.markdown(
        """
        <div class="panel-title">Manual customer scoring</div>
        <div class="panel-caption">
            Enter the customer profile and run a single prediction. Inputs are grouped to keep the workflow calm and readable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("manual_prediction_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.subheader("Customer profile")
            tenure = st.number_input("Tenure in months", min_value=0, max_value=100, value=12, step=1)
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=75.0,
                step=1.0,
            )
            referrals = st.number_input("Number of referrals", min_value=0, max_value=50, value=1, step=1)
            married = st.selectbox("Married", ["No", "Yes"])

        with col_b:
            st.subheader("Plan and services")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            )
            online_security = st.selectbox("Online security", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        customer = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Number of Referrals": referrals,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
            "OnlineSecurity": online_security,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Age": age,
            "Married": married,
        }
        customer_df = pd.DataFrame([customer])
        probability, cluster = predict_churn(customer_df, artifacts)
        probability = float(probability[0])
        cluster = int(cluster[0])
        risk = risk_level(probability)
        revenue = revenue_at_risk(probability, monthly_charges)
        saved_df = customer_df.copy()
        saved_df["Churn Probability"] = probability
        saved_df["Risk Level"] = risk
        saved_df["Cluster"] = cluster
        saved_df["Revenue at Risk"] = revenue
        save_prediction_records(saved_df, "Manual")

        render_section_heading(
            "Result",
            "Churn probability, segment, and revenue exposure for the submitted customer.",
        )
        with st.container(border=True):
            render_single_prediction_result(probability, risk, cluster, revenue)

        render_section_heading(
            "Customer summary",
            "Key profile and service details used by the model for this prediction.",
        )
        with st.container(border=True):
            render_customer_summary(customer)


def style_results_table(df):
    preferred = [
        "tenure",
        "MonthlyCharges",
        "Age",
        "Contract",
        "InternetService",
        "Risk Level",
        "Churn Probability",
        "Cluster",
        "Revenue at Risk",
    ]
    columns_to_show = [column for column in preferred if column in df.columns]
    remaining = [column for column in df.columns if column not in columns_to_show]
    table_df = df[columns_to_show + remaining].copy()

    formatters = {}
    if "Churn Probability" in table_df.columns:
        formatters["Churn Probability"] = "{:.1%}"
    if "Revenue at Risk" in table_df.columns:
        formatters["Revenue at Risk"] = "${:,.2f}"
    if "MonthlyCharges" in table_df.columns:
        formatters["MonthlyCharges"] = "${:,.2f}"

    def color_risk(value):
        colors = {
            "High": "color: #fb7185; font-weight: 700;",
            "Medium": "color: #fbbf24; font-weight: 700;",
            "Low": "color: #34d399; font-weight: 700;",
        }
        return colors.get(value, "")

    styler = table_df.style.format(formatters)
    if "Risk Level" in table_df.columns:
        styler = styler.applymap(color_risk, subset=["Risk Level"])
    return styler


def render_csv_upload(artifacts):
    st.markdown(
        """
        <div class="panel-title">CSV batch scoring</div>
        <div class="panel-caption">
            Upload a customer CSV with the same business fields used by the model. Monthly charges should be in USD. Results appear as a clean table with risk labels and export support.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to generate batch predictions.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    if raw_df.empty:
        st.warning("The uploaded CSV has no rows to score.")
        return

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in raw_df.columns]
    if missing:
        st.warning(
            "Some recommended columns are missing. The app will score what it can and fill missing model inputs with zeros: "
            + ", ".join(missing)
        )

    result_df = build_prediction_frame(raw_df, artifacts)
    high_count = int((result_df["Risk Level"] == "High").sum())
    avg_probability = float(result_df["Churn Probability"].mean())
    revenue_total = float(result_df["Revenue at Risk"].sum())

    render_section_heading(
        "Batch result",
        "A quick read on the imported file before drilling into charts and customer rows.",
    )
    render_metric_grid(
        [
            ("Customers Scored", f"{len(result_df):,}", ""),
            ("Average Churn Probability", f"{avg_probability:.1%}", ""),
            ("High Risk Customers", f"{high_count:,}", "risk-high"),
            ("Total Revenue at Risk", format_currency(revenue_total), ""),
        ],
        columns=4,
    )

    render_section_heading(
        "Risk charts",
        "Distribution and probability spread for the uploaded customer set.",
    )
    chart_col_1, chart_col_2 = st.columns(2, gap="large")
    with chart_col_1:
        with st.container(border=True):
            render_chart_header(
                "Risk distribution",
                "Customer count split across low, medium, and high churn risk bands.",
            )
            render_plotly_chart(risk_distribution_chart(result_df["Risk Level"]), height=305, show_legend=True)
    with chart_col_2:
        with st.container(border=True):
            render_chart_header(
                "Probability spread",
                "Distribution of predicted churn probabilities with medium and high risk thresholds.",
            )
            render_plotly_chart(probability_histogram(result_df["Churn Probability"]), height=305)

    render_section_heading(
        "Scored customers",
        "Readable output table with churn probability, risk level, cluster, and revenue at risk.",
    )
    st.dataframe(style_results_table(result_df), use_container_width=True, hide_index=True, height=420)
    save_col, download_col = st.columns(2)
    with save_col:
        if st.button("Save batch to database"):
            save_prediction_records(result_df, "CSV")
            st.success(f"Saved {len(result_df):,} predictions to the local database.")
    with download_col:
        st.download_button(
            "Download scored CSV",
            result_df.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )


def render_analytics():
    render_page_intro(
        "Saved customer intelligence",
        "Analytics from stored predictions.",
        "Every manual prediction and saved CSV batch can be reviewed here. Use it to track risk mix, cluster behavior, and revenue exposure over time.",
    )

    records = load_saved_predictions()
    if records.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-title">No saved predictions yet</div>
                <div class="panel-caption">
                    Run a manual prediction or save a CSV batch from the Predict page. Your saved records will appear here.
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    records["created_at"] = pd.to_datetime(records["created_at"], errors="coerce")
    total_revenue = float(records["revenue_at_risk"].sum())
    average_probability = float(records["churn_probability"].mean())
    high_risk_count = int((records["risk_level"] == "High").sum())

    render_section_heading(
        "Saved prediction summary",
        "Current database totals across manual predictions and saved CSV imports.",
        first=True,
    )
    render_metric_grid(
        [
            ("Saved Predictions", f"{len(records):,}", ""),
            ("Average Churn Probability", f"{average_probability:.1%}", ""),
            ("High Risk Records", f"{high_risk_count:,}", "risk-high"),
            ("Revenue Exposure", format_currency(total_revenue), ""),
        ],
        columns=4,
    )

    render_section_heading(
        "Analytics",
        "Model outputs visualized with consistent spacing across risk, clusters, revenue, and sources.",
    )
    chart_col_a, chart_col_b = st.columns(2, gap="medium")
    with chart_col_a:
        with st.container(border=True):
            render_chart_header(
                "Risk distribution",
                "Saved predictions grouped by churn risk band.",
            )
            render_plotly_chart(risk_distribution_chart(records["risk_level"]), height=300)

    with chart_col_b:
        with st.container(border=True):
            render_chart_header(
                "Average risk by cluster",
                "Mean churn probability for each KMeans customer segment.",
            )
            render_plotly_chart(cluster_risk_chart(records), height=300)

    with st.container(border=True):
        render_chart_header(
            "Revenue at risk by band",
            "Estimated monthly revenue exposure by churn risk group.",
        )
        render_plotly_chart(revenue_by_risk_chart(records), height=300)

    render_section_heading(
        "Probability distribution",
        "The overall spread of saved predictions across the 0-100 percent risk range.",
    )
    with st.container(border=True):
        render_chart_header(
            "Churn probability spread",
            "The overall distribution of saved predictions across the 0-100 percent risk range.",
        )
        render_plotly_chart(probability_histogram(records["churn_probability"]), height=320)

    render_section_heading(
        "Recent records",
        "Latest saved predictions from the local SQLite database.",
    )
    with st.container(border=True):
        st.markdown('<div class="panel-title">Recent saved records</div>', unsafe_allow_html=True)
        render_recent_records_table(records)


def render_predict(artifacts):
    render_page_intro(
        "Prediction workspace",
        "Score churn risk without the noise.",
        "Choose a single customer form or a CSV upload. Both paths use the same saved preprocessing columns, scaler, ANN model, and KMeans cluster model.",
    )

    manual_tab, csv_tab = st.tabs(["Manual Input", "CSV Upload"])
    with manual_tab:
        with st.container(border=True):
            render_manual_input(artifacts)

    with csv_tab:
        with st.container(border=True):
            render_csv_upload(artifacts)


def main():
    inject_css()
    init_database()
    artifacts = load_artifacts()
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Predict":
        render_predict(artifacts)
    else:
        render_analytics()


if __name__ == "__main__":
    main()
