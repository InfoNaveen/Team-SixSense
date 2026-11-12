"""
app.py — Delhi DISCOM Load Forecasting & Procurement Copilot Dashboard
======================================================================
Reads results/ files ONLY. Never imports models/, features/, or data/.
Runs 100% offline.

Theme: Editorial Luxury (Financial Times / High-End Financial Research)
- Canvas: Cream (--cream: #FAF8F4)
- Surfaces / Cards: Warm Surface (--surface: #F2EDE6)
- Typography: Headings in 'Playfair Display', Body in 'DM Sans', Data/Metrics in 'JetBrains Mono'
- Strict Rules: Zero gradients, zero blurs, 1px solid #E8E3DC borders, 8px/4px radii.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="Delhi DISCOM Copilot | Research Edition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

# ==============================================================================
# 1. EDITORIAL LUXURY COLOR TOKENS
# ==============================================================================

COLOR_CREAM = "#FAF8F4"        # Page Background (--cream)
COLOR_SURFACE = "#F2EDE6"      # Card / Surface Background (--surface)
COLOR_INK = "#1A1814"          # Primary Text (--ink)
COLOR_INK_MUTED = "#6B6660"    # Secondary / Muted Text & Labels (--ink-muted)
COLOR_GOLD = "#C9A84C"         # Primary Accent / Key Metrics / CTA (--gold)
COLOR_GOLD_LIGHT = "#E8D5A3"   # Subtle Gold Highlights & Fills (--gold-light)
COLOR_BORDER = "#E8E3DC"       # Borders (--border)
COLOR_RISK_RED = "#C0392B"     # Danger / High Risk / Losses (--risk-red)
COLOR_RISK_GREEN = "#1A6B3C"   # Safe / Low Risk / Gains (--risk-green)
COLOR_SLATE_GRAY = "#C2BCB3"   # Editorial warm gray for baseline comparison

# Unified Model Colours across all bar charts & graphs
MODEL_COLORS = {
    "lgbm_quantile": COLOR_GOLD,             # Gold (#C9A84C)
    "lgbm_quantile_calibrated": COLOR_GOLD_LIGHT, # Subtle Gold (#E8D5A3)
    "linear_regression": COLOR_INK_MUTED,    # Muted Ink (#6B6660)
    "seasonal_naive": COLOR_SLATE_GRAY,      # Warm Editorial Gray (#C2BCB3)
}

# Standard Plotly Colorway for luxury editorial publications
PLOTLY_COLORWAY = [
    COLOR_GOLD,
    COLOR_GOLD_LIGHT,
    COLOR_INK_MUTED,
    COLOR_SLATE_GRAY,
    COLOR_RISK_RED,
    COLOR_RISK_GREEN,
]

# ==============================================================================
# 2. EDITORIAL LUXURY CUSTOM CSS
# ==============================================================================

EDITORIAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&family=JetBrains+Mono:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap');

:root {{
    --cream: {COLOR_CREAM};
    --surface: {COLOR_SURFACE};
    --ink: {COLOR_INK};
    --ink-muted: {COLOR_INK_MUTED};
    --gold: {COLOR_GOLD};
    --gold-light: {COLOR_GOLD_LIGHT};
    --border: {COLOR_BORDER};
    --risk-red: {COLOR_RISK_RED};
    --risk-green: {COLOR_RISK_GREEN};
}}

/* Base application canvas & text */
.stApp {{
    background-color: var(--cream) !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

/* Headings — Playfair Display Serif Authority */
h1, h2, h3, h4, h5, h6, .stHeading {{
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}}

h1 {{
    font-size: 2.3rem !important;
    line-height: 1.18 !important;
}}

h2, h3 {{
    font-size: 1.35rem !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 6px !important;
    margin-top: 18px !important;
}}

/* Sidebar styling — Solid Surface with Sharp Structured 1px Border */
section[data-testid="stSidebar"] {{
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {{
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--ink) !important;
}}

/* KPI Metric Cards — Crisp Flat Surface, 1px Border, 8px Radius, JetBrains Mono Numbers */
div[data-testid="stMetric"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease !important;
}}

div[data-testid="stMetric"]:hover {{
    border-color: var(--gold) !important;
}}

div[data-testid="stMetricLabel"] p {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--ink-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 4px !important;
}}

div[data-testid="stMetricValue"] div {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    letter-spacing: -0.02em !important;
}}

div[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
}}

/* Tab Navigation — Flat Crisp Tabs with Gold Active Border */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px !important;
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0px !important;
}}

.stTabs [data-baseweb="tab"] {{
    height: 40px !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 8px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    color: var(--ink-muted) !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    transition: all 0.15s ease !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: var(--ink) !important;
    background-color: var(--surface) !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-bottom: 2px solid var(--gold) !important;
}}

/* Top Hero Publication Header */
.editorial-header {{
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-top: 4px solid var(--gold);
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
}}

.badge-tag {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid var(--border);
}}

.badge-primary {{ background-color: var(--ink); color: var(--cream); border-color: var(--ink); }}
.badge-gold {{ background-color: var(--gold-light); color: var(--ink); border-color: var(--gold); }}
.badge-green {{ background-color: #E6EFEA; color: var(--risk-green); border-color: #A3CBB6; }}

/* Big Single Number Class (54pt) in JetBrains Mono */
.big-number-saving {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 50pt;
    font-weight: 800;
    color: var(--risk-green);
    line-height: 1.05;
    letter-spacing: -0.03em;
}}

/* Dataframe & Tables Styling */
div[data-testid="stDataFrame"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}}

div[data-testid="stDataFrame"] table {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}}

/* Inputs, Selectboxes, Radio Buttons — 4px Radius, 1px Border */
div[data-baseweb="select"] > div, 
input, select, textarea {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* Code / Monospace elements */
code, pre {{
    font-family: 'JetBrains Mono', monospace !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--ink) !important;
    padding: 2px 5px !important;
}}
</style>
"""

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ==============================================================================
# DATA LOADERS (results/ ONLY)
# ==============================================================================

def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter (DJF)"
    if month in (3, 4, 5, 6):
        return "Summer (MAMJ)"
    if month in (7, 8, 9):
        return "Monsoon (JAS)"
    return "Post-Monsoon (ON)"


@st.cache_data
def load_forecasts(horizon: str) -> pd.DataFrame:
    csv_path = RESULTS_DIR / f"forecasts_{horizon}.csv"
    pq_path = RESULTS_DIR / f"forecasts_{horizon}.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif pq_path.exists():
        df = pd.read_parquet(pq_path)
    else:
        st.error(f"Missing {csv_path.name}")
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_peak_flags() -> pd.DataFrame:
    csv_path = RESULTS_DIR / "peak_flags.csv"
    pq_path = RESULTS_DIR / "peak_flags.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif pq_path.exists():
        return pd.read_parquet(pq_path)
    return pd.DataFrame()


@st.cache_data
def load_missed_peaks() -> pd.DataFrame:
    path = RESULTS_DIR / "missed_peaks.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_procurement() -> dict[str, Any]:
    path = RESULTS_DIR / "procurement_week.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ==============================================================================
# TAB 1: FORECAST (Crisp Editorial Hero Chart)
# ==============================================================================

def render_forecast_tab(df: pd.DataFrame, peak_flags: pd.DataFrame, horizon: str) -> None:
    st.subheader(f"Probabilistic Demand Forecast — {horizon.upper()}-Ahead Evaluation")

    if df.empty:
        st.warning("No forecast data available.")
        return

    work = df.copy().sort_values("ts")

    c1, c2 = st.columns([1, 1])
    models = sorted(work["model"].dropna().unique().tolist())
    def_idx = 0
    for i, m in enumerate(models):
        if "calibrated" in m:
            def_idx = i
            break
        elif "lgbm" in m:
            def_idx = i

    sel_model = c1.selectbox("Forecast Model Architecture", models, index=def_idx, key=f"sel_model_{horizon}")
    work = work[work["model"] == sel_model]

    folds = sorted(work["fold"].dropna().unique().tolist())
    sel_fold = c2.selectbox("Walk-Forward Test Fold", ["All Folds"] + [str(f) for f in folds], key=f"sel_fold_{horizon}")
    if sel_fold != "All Folds":
        work = work[work["fold"].astype(str) == sel_fold]

    # KPIs in Monospace
    inside_band = ((work["actual"] >= work["p10"]) & (work["actual"] <= work["p90"])).mean() * 100.0
    rmse = float(np.sqrt(np.mean((work["actual"] - work["p50"]) ** 2)))
    peak_act = float(work["actual"].max())

    k1, k2, k3 = st.columns(3)
    k1.metric("Peak Observed Load", f"{peak_act:,.0f} MW")
    k2.metric("Point Forecast RMSE", f"{rmse:,.1f} MW")
    picp_delta = f"{inside_band - 80.0:+.1f}% vs 80% Target"
    k3.metric("Empirical PICP (P10–P90)", f"{inside_band:.1f}%", picp_delta)

    # Plotly Chart — Solid Crisp Surface, Zero Gradients, Sharp Monospace Ticks
    fig = go.Figure()

    # P90 upper bound line (thin gold boundary)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p90"], mode="lines",
        line=dict(color="rgba(201, 168, 76, 0.6)", width=1.0),
        name="P90 Upper Bound", showlegend=False
    ))

    # P10 lower bound line with subtle gold-light fill (--gold-light: #E8D5A3 at 28%)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p10"], mode="lines",
        line=dict(color="rgba(201, 168, 76, 0.6)", width=1.0),
        fill="tonexty",
        fillcolor="rgba(232, 213, 163, 0.32)",
        name="P10–P90 Confidence Interval",
    ))

    # P50 point forecast (Primary Accent Gold #C9A84C dashed line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p50"], mode="lines",
        line=dict(color=COLOR_GOLD, width=2.4, dash="dash"),
        name="P50 Expected Forecast",
    ))

    # Actual demand (Deep Ink #1A1814 solid line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["actual"], mode="lines",
        line=dict(color=COLOR_INK, width=2.2),
        name="Actual Grid Demand (MW)",
    ))

    # Peak markers
    if not peak_flags.empty:
        pf = peak_flags.copy()
        pf["date_str"] = pd.to_datetime(pf["date"]).dt.strftime("%Y-%m-%d")
        work["date_str"] = work["ts"].dt.strftime("%Y-%m-%d")
        work["hour"] = work["ts"].dt.hour

        pred_peaks = pf[pf["peak_pred"].astype(bool)]
        if not pred_peaks.empty:
            merged = work.merge(
                pred_peaks[["date_str", "peak_hour_pred", "peak_prob"]],
                left_on=["date_str", "hour"], right_on=["date_str", "peak_hour_pred"], how="inner"
            )
            if not merged.empty:
                fig.add_trace(go.Scatter(
                    x=merged["ts"], y=merged["p90"], mode="markers",
                    marker=dict(color=COLOR_GOLD, size=9, symbol="diamond", line=dict(color=COLOR_INK, width=1.2)),
                    name="Predicted Peak Hour",
                ))

        actual_peaks = pf[pf["peak_actual"].astype(bool)]
        if not actual_peaks.empty:
            merged_act = work.merge(
                actual_peaks[["date_str", "peak_hour_actual"]],
                left_on=["date_str", "hour"], right_on=["date_str", "peak_hour_actual"], how="inner"
            )
            if not merged_act.empty:
                fig.add_trace(go.Scatter(
                    x=merged_act["ts"], y=merged_act["actual"], mode="markers",
                    marker=dict(color=COLOR_RISK_RED, size=10, symbol="circle", line=dict(color=COLOR_CREAM, width=1.5)),
                    name="Actual Peak Event (>90th %ile)",
                ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_CREAM,
        colorway=PLOTLY_COLORWAY,
        height=470,
        margin=dict(l=24, r=24, t=36, b=24),
        font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
        xaxis=dict(
            title=dict(text="Time (IST)", font=dict(color=COLOR_INK, size=12, family="DM Sans, sans-serif")),
            showgrid=True,
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_INK_MUTED, size=11, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            title=dict(text="Demand (MW)", font=dict(color=COLOR_INK, size=12, family="DM Sans, sans-serif")),
            showgrid=True,
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_INK_MUTED, size=11, family="JetBrains Mono, monospace"),
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=COLOR_SURFACE,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(size=11, color=COLOR_INK, family="DM Sans, sans-serif"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: ACCURACY VS BASELINES (Structured Editorial Bar Charts)
# ==============================================================================

def render_accuracy_tab(metrics: pd.DataFrame, horizon: str) -> None:
    st.subheader(f"Walk-Forward Comparative Evaluation — {horizon.upper()}-Ahead")
    if metrics.empty:
        st.info("No metrics available.")
        return

    m_sub = metrics[metrics["horizon"] == horizon].copy()
    avg_metrics = m_sub.groupby(["model", "metric"], as_index=False)["value"].mean()
    pivoted = avg_metrics.pivot(index="model", columns="metric", values="value").round(2)

    st.dataframe(pivoted, use_container_width=True)

    # Bar charts comparing MAPE & RMSE with JetBrains Mono numbers
    piv_reset = pivoted.reset_index()
    if "MAPE" in piv_reset.columns and "RMSE" in piv_reset.columns:
        c1, c2 = st.columns(2)

        with c1:
            fig_mape = px.bar(
                piv_reset, x="model", y="MAPE",
                color="model",
                color_discrete_map=MODEL_COLORS,
                title="<b>MAPE (%)</b> — Lower is better",
                text="MAPE",
            )
            fig_mape.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                textfont=dict(color=COLOR_INK, size=11, family="JetBrains Mono, monospace"),
                marker_line_width=1,
                marker_line_color=COLOR_BORDER,
            )
            fig_mape.update_layout(
                template="plotly_white",
                paper_bgcolor=COLOR_SURFACE,
                plot_bgcolor=COLOR_CREAM,
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
                xaxis=dict(title="", tickfont=dict(size=11, color=COLOR_INK)),
                yaxis=dict(title="MAPE (%)", gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_INK_MUTED, family="JetBrains Mono, monospace")),
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        with c2:
            fig_rmse = px.bar(
                piv_reset, x="model", y="RMSE",
                color="model",
                color_discrete_map=MODEL_COLORS,
                title="<b>RMSE (MW)</b> — Lower is better",
                text="RMSE",
            )
            fig_rmse.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
                textfont=dict(color=COLOR_INK, size=11, family="JetBrains Mono, monospace"),
                marker_line_width=1,
                marker_line_color=COLOR_BORDER,
            )
            fig_rmse.update_layout(
                template="plotly_white",
                paper_bgcolor=COLOR_SURFACE,
                plot_bgcolor=COLOR_CREAM,
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
                xaxis=dict(title="", tickfont=dict(size=11, color=COLOR_INK)),
                yaxis=dict(title="RMSE (MW)", gridcolor=COLOR_BORDER, tickfont=dict(color=COLOR_INK_MUTED, family="JetBrains Mono, monospace")),
            )
            st.plotly_chart(fig_rmse, use_container_width=True)


# ==============================================================================
# TAB 3: ERROR HEATMAP (Warm Editorial Research Scale)
# ==============================================================================

def render_heatmap_tab(df: pd.DataFrame) -> None:
    st.subheader("Hour × Season Dispersion Heatmap (Mean Absolute Error, MW)")
    if df.empty:
        st.info("No forecast data available.")
        return

    f_sub = df.copy()
    if "model" in f_sub.columns:
        pref = [m for m in f_sub["model"].unique() if "lgbm" in m]
        if pref:
            f_sub = f_sub[f_sub["model"] == pref[0]]

    f_sub["hour"] = f_sub["ts"].dt.hour
    f_sub["season"] = f_sub["ts"].dt.month.map(_season_from_month)
    f_sub["abs_err"] = (f_sub["actual"] - f_sub["p50"]).abs()

    season_order = ["Winter (DJF)", "Summer (MAMJ)", "Monsoon (JAS)", "Post-Monsoon (ON)"]
    heat = f_sub.pivot_table(index="season", columns="hour", values="abs_err", aggfunc="mean").reindex(
        [s for s in season_order if s in f_sub["season"].unique()]
    )

    # Warm Editorial Publication Scale: Cream -> Gold Light -> Gold -> Deep Ink
    editorial_heat_scale = [
        [0.0, COLOR_CREAM],
        [0.25, COLOR_SURFACE],
        [0.55, COLOR_GOLD_LIGHT],
        [0.85, COLOR_GOLD],
        [1.0, COLOR_INK],
    ]

    fig = px.imshow(
        heat,
        labels=dict(x="Hour of Day (IST)", y="Season", color="MAE (MW)"),
        color_continuous_scale=editorial_heat_scale,
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLOR_SURFACE,
        plot_bgcolor=COLOR_CREAM,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_INK_MUTED, size=11, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            tickfont=dict(color=COLOR_INK, size=12, weight="bold"),
        ),
        coloraxis_colorbar=dict(
            title=dict(text="MAE (MW)", font=dict(size=11, color=COLOR_INK)),
            tickfont=dict(size=10, color=COLOR_INK_MUTED, family="JetBrains Mono, monospace"),
            outlinecolor=COLOR_BORDER,
            outlinewidth=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-left: 3px solid {COLOR_GOLD}; padding: 14px 20px; border-radius: 8px; font-size: 13px; color: {COLOR_INK};">
            <b>Structural Finding:</b> Peak forecast divergence heavily concentrates in Summer (MAMJ) evening hours (19:00 – 23:00 IST), induced by non-linear residential air-conditioning loads and nocturnal heat trapping.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 4: MISSED PEAKS (Editorial Risk Diagnostic)
# ==============================================================================

def render_missed_peaks_tab(missed: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Empirical Failure Analysis: Uncaptured Peak Events")
    if missed.empty:
        st.success("No peak events were missed by the P90 threshold detector!")
        return

    st.markdown(f"**Days where actual grid demand breached top-decile threshold without P90 trigger:**")
    st.dataframe(missed, use_container_width=True, hide_index=True)

    dates = missed["date"].astype(str).tolist()
    sel_date = st.selectbox("Inspect Specific Failure Event", dates)
    row = missed[missed["date"].astype(str) == sel_date].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Actual Peak Load", f"{row['actual_peak_mw']:,.0f} MW")
    k2.metric("Predicted P50", f"{row['pred_p50_peak_mw']:,.0f} MW")
    k3.metric("Predicted P90", f"{row['pred_p90_peak_mw']:,.0f} MW")
    k4.metric("Temporal Phase Error", f"{row['hour_error_min']:,.0f} min")

    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-left: 3px solid {COLOR_RISK_RED}; padding: 14px 20px; border-radius: 8px; font-size: 13px; color: {COLOR_INK}; margin: 16px 0;">
            <b style="color: {COLOR_RISK_RED};">Diagnostic Note ({sel_date}):</b> {row.get('note', 'Sudden thermal/weather shift')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Zoom in on that day
    f_sub = df.copy()
    f_sub["date_str"] = f_sub["ts"].dt.strftime("%Y-%m-%d")
    day_slice = f_sub[f_sub["date_str"] == sel_date].sort_values("ts")
    if not day_slice.empty:
        pref = [m for m in day_slice["model"].unique() if "lgbm" in m]
        if pref:
            day_slice = day_slice[day_slice["model"] == pref[0]]

        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p90"], mode="lines",
            line=dict(color="rgba(192, 57, 43, 0.5)", width=1.0),
            name="P90 Upper Bound", showlegend=False
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p10"], mode="lines",
            line=dict(color="rgba(192, 57, 43, 0.5)", width=1.0),
            fill="tonexty",
            fillcolor="rgba(192, 57, 43, 0.15)",
            name="P10–P90 Missed Range",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p50"], mode="lines",
            line=dict(dash="dash", color=COLOR_GOLD, width=2.4),
            name="P50 Forecast",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["actual"], mode="lines+markers",
            line=dict(color=COLOR_RISK_RED, width=2.2),
            marker=dict(size=6, color=COLOR_RISK_RED, line=dict(color=COLOR_CREAM, width=1)),
            name="Actual Demand (Peak Surge)",
        ))

        fig_day.update_layout(
            template="plotly_white",
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_CREAM,
            height=340,
            margin=dict(l=20, r=20, t=24, b=20),
            font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
            yaxis_title="Load (MW)",
            xaxis_title="Time of Day (IST)",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor=COLOR_SURFACE, bordercolor=COLOR_BORDER, borderwidth=1
            ),
        )
        st.plotly_chart(fig_day, use_container_width=True)


# ==============================================================================
# TAB 5: PROCUREMENT COPILOT (Editorial Financial Optimization)
# ==============================================================================

def render_procurement_tab(proc: dict) -> None:
    st.subheader("Procurement Copilot — Risk-Aware Dispatch Optimization")
    if not proc:
        st.warning("No procurement results available.")
        return

    c_p90 = proc.get("cost_p90", proc.get("cost_model", 0.0)) / 1e7
    c_flat = proc.get("cost_flat", 0.0) / 1e7
    c_p50 = proc.get("cost_p50", 0.0) / 1e7
    saving = proc.get("saving_p90_vs_flat", proc.get("saving", 0.0)) / 1e7
    pct = proc.get("saving_pct", proc.get("saving_pct_vs_flat", 0.0))
    exposure = proc.get("p90_exposure_mw", 0.0)

    # Hero Rupee Savings Banner — Flat Surface, 1px Border, 8px Radius, Big Monospace Saving
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; border-left: 4px solid {COLOR_RISK_GREEN}; border-radius: 8px; padding: 24px 28px; margin-bottom: 24px;">
            <div style="font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 700; color: {COLOR_RISK_GREEN}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                Representative Peak Dispatch Window (Fold 6 — 168 Hours)
            </div>
            <div class="big-number-saving">₹{saving:,.2f} Cr</div>
            <div style="font-family: 'DM Sans', sans-serif; font-size: 16px; font-weight: 600; color: {COLOR_INK}; margin-top: 4px;">
                Net Rupee Savings (+{pct:.2f}% Cost Reduction vs Flat Contracted Baseline)
            </div>
            <div style="font-family: 'DM Sans', sans-serif; font-size: 13px; color: {COLOR_INK_MUTED}; margin-top: 8px;">
                Dynamic P90 scheduling prevents unhedged exposure to punitive ₹12.00/kWh DSM deviation penalties while mitigating stranded capacity risk.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P90 Dynamic Schedule", f"₹{c_p90:,.2f} Cr")
    k2.metric("P50 Median Schedule", f"₹{c_p50:,.2f} Cr")
    k3.metric("Flat Baseline Schedule", f"₹{c_flat:,.2f} Cr")
    k4.metric("Max P90 Risk Exposure", f"{exposure:,.1f} MW")

    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; padding: 12px 18px; border-radius: 4px; font-size: 12px; color: {COLOR_INK_MUTED}; margin: 16px 0; font-family: 'DM Sans', sans-serif;">
            <b>Tariff Framework (Marked Assumptions):</b> Base Contracted Rate = ₹5.00/kWh | DSM Shortfall Penalty = ₹12.00/kWh | Stranded Surplus Loss = ₹2.50/kWh
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hourly detail chart
    if "hourly_detail_p90" in proc and len(proc["hourly_detail_p90"]) > 0:
        h_df = pd.DataFrame(proc["hourly_detail_p90"])
        h_df["ts"] = pd.to_datetime(h_df["ts"])

        st.markdown("**Hourly Dispatch Schedule vs Actual Load (Fold 6 — 168h)**")
        fig_sched = go.Figure()

        # Flat naive schedule (Muted Slate dashed line)
        if "schedule_flat" in proc:
            flat_val = float(proc["schedule_flat"][0]) if isinstance(proc["schedule_flat"], list) else float(proc["schedule_flat"])
            fig_sched.add_trace(go.Scatter(
                x=h_df["ts"], y=[flat_val] * len(h_df), mode="lines",
                line=dict(color=COLOR_INK_MUTED, width=1.8, dash="dot"),
                name="Flat Baseline Schedule",
            ))

        # Dynamic P90 Schedule (Risk Green #1A6B3C)
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["schedule_mw"], mode="lines",
            line=dict(color=COLOR_RISK_GREEN, width=2.4),
            name="Dynamic P90 Schedule",
        ))

        # Actual Demand (Deep Ink #1A1814)
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["actual_mw"], mode="lines",
            line=dict(color=COLOR_INK, width=2.0),
            name="Actual Demand (MW)",
        ))

        fig_sched.update_layout(
            template="plotly_white",
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_CREAM,
            height=370,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="DM Sans, sans-serif", color=COLOR_INK),
            yaxis=dict(
                title=dict(text="Load (MW)", font=dict(color=COLOR_INK, size=12, family="DM Sans, sans-serif")),
                gridcolor=COLOR_BORDER,
                linecolor=COLOR_BORDER,
                tickfont=dict(color=COLOR_INK_MUTED, size=11, family="JetBrains Mono, monospace"),
            ),
            xaxis=dict(
                title=dict(text="Time (IST)", font=dict(color=COLOR_INK, size=12, family="DM Sans, sans-serif")),
                gridcolor=COLOR_BORDER,
                linecolor=COLOR_BORDER,
                tickfont=dict(color=COLOR_INK_MUTED, size=11, family="JetBrains Mono, monospace"),
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=COLOR_SURFACE,
                bordercolor=COLOR_BORDER,
                borderwidth=1,
                font=dict(size=11, color=COLOR_INK, family="DM Sans, sans-serif"),
            ),
        )
        st.plotly_chart(fig_sched, use_container_width=True)

    # Sensitivity Table
    if "sensitivity_table" in proc:
        st.markdown("**Tariff Sensitivity Analysis (DSM Penalty vs Stranded Surplus Rates)**")
        sens_df = pd.DataFrame(proc["sensitivity_table"])
        st.dataframe(sens_df, use_container_width=True, hide_index=True)


# ==============================================================================
# MAIN APP ENTRYPOINT
# ==============================================================================

def main():
    # Top Editorial Publication Header
    st.markdown(
        f"""
        <div class="editorial-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span class="badge-tag badge-primary">DELHI DISCOM</span>
                        <span class="badge-tag badge-gold">QUANTILE RESEARCH</span>
                        <span class="badge-tag badge-green">OPTIMIZED DISPATCH</span>
                    </div>
                    <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 27px; font-weight: 800; color: {COLOR_INK}; margin: 0; letter-spacing: -0.01em;">
                        Probabilistic Peak-Aware Load Forecasting & Procurement Copilot
                    </div>
                    <div style="font-family: 'DM Sans', sans-serif; font-size: 13px; color: {COLOR_INK_MUTED}; margin-top: 5px;">
                        Hourly P10/P50/P90 Trajectories • Conformal Walk-Forward Calibration • Risk-Constrained Hedging
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### ⚙️ **Control Deck**")
        horizon = st.radio(
            "Forecast Horizon",
            ["day", "week"],
            index=0,
            format_func=lambda h: "Day-Ahead (24h)" if h == "day" else "Week-Ahead (168h)"
        )

        st.divider()
        st.markdown(f"### 📋 **Interface Verification**")
        for f in ["forecasts_day.csv", "forecasts_week.csv", "metrics.csv", "peak_flags.csv", "missed_peaks.csv", "procurement_week.json"]:
            p = RESULTS_DIR / f
            if p.exists():
                st.markdown(f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 12px; margin-bottom: 5px;'><span style='color: {COLOR_RISK_GREEN}; font-weight: bold;'>✓</span> <code>{f}</code></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 12px; margin-bottom: 5px;'><span style='color: {COLOR_RISK_RED}; font-weight: bold;'>✗</span> <code>{f}</code></div>", unsafe_allow_html=True)

        st.divider()
        st.caption("🔒 Financial research edition reading verified `results/` contracts.")

    forecasts_df = load_forecasts(horizon)
    metrics_df = load_metrics()
    peak_flags_df = load_peak_flags()
    missed_peaks_df = load_missed_peaks()
    proc_dict = load_procurement()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Forecast",
        "Accuracy vs baselines",
        "Error heatmap",
        "Missed peaks",
        "Procurement",
    ])

    with tab1:
        render_forecast_tab(forecasts_df, peak_flags_df, horizon)
    with tab2:
        render_accuracy_tab(metrics_df, horizon)
    with tab3:
        render_heatmap_tab(forecasts_df)
    with tab4:
        render_missed_peaks_tab(missed_peaks_df, forecasts_df)
    with tab5:
        render_procurement_tab(proc_dict)


if __name__ == "__main__":
    main()
