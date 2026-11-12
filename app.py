"""
app.py — Delhi DISCOM Load Forecasting & Procurement Copilot Dashboard
======================================================================
Reads results/ files ONLY. Never imports models/, features/, or data/.
Runs 100% offline.

Theme: Official Brand Guidelines
- Primary: Deep Navy (#12355B)
- Forecast: Signal Blue (#4682DC)
- Band: Blue Tint (#4682DC @ 25% opacity, #D9E6F9)
- Actual: Ink (#111827)
- Peak / Risk: Peak Red (#D64545)
- Saving: Grid Green (#2E9E6B)
- Baseline: Slate (#8A94A6) / Lighter Slate (#C3CAD6)
- Highlight: Amber (#F2A93B)
- Background: White (#FFFFFF)
- Surface: Mist (#F3F6FA)

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
    page_title="Delhi DISCOM Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"

# ==============================================================================
# BRAND PALETTE CONSTANTS
# ==============================================================================

COLOR_PRIMARY = "#12355B"    # Deep Navy
COLOR_FORECAST = "#4682DC"   # Signal Blue
COLOR_BAND = "rgba(70, 130, 220, 0.25)"  # Blue Tint (25% opacity)
COLOR_BAND_HEX = "#D9E6F9"   # Blue Tint on white
COLOR_ACTUAL = "#111827"     # Ink
COLOR_PEAK = "#D64545"       # Peak Red
COLOR_SAVING = "#2E9E6B"     # Grid Green
COLOR_BASELINE = "#8A94A6"   # Slate
COLOR_NAIVE = "#C3CAD6"      # Lighter Slate
COLOR_HIGHLIGHT = "#F2A93B"  # Amber
COLOR_BG = "#FFFFFF"         # White
COLOR_SURFACE = "#F3F6FA"    # Mist

# Unified Model Colours (Identical across all charts & tabs)
MODEL_COLORS = {
    "lgbm_quantile": COLOR_FORECAST,             # Signal Blue
    "lgbm_quantile_calibrated": COLOR_HIGHLIGHT, # Amber
    "linear_regression": COLOR_BASELINE,         # Slate
    "seasonal_naive": COLOR_NAIVE,               # Lighter Slate
}

# Standard Plotly Colorway
PLOTLY_COLORWAY = [
    COLOR_FORECAST,   # #4682DC
    COLOR_HIGHLIGHT,  # #F2A93B
    COLOR_BASELINE,   # #8A94A6
    COLOR_NAIVE,      # #C3CAD6
    COLOR_PEAK,       # #D64545
    COLOR_SAVING,     # #2E9E6B
]

# ==============================================================================
# BRAND THEME CUSTOM CSS
# ==============================================================================

CUSTOM_BRAND_CSS = f"""
<style>
/* Base application typography & canvas */
.stApp {{
    background-color: {COLOR_BG} !important;
    color: {COLOR_PRIMARY} !important;
    font-family: Calibri, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6, .stHeading {{
    color: {COLOR_PRIMARY} !important;
    font-family: Calibri, Inter, sans-serif !important;
    font-weight: 700 !important;
}}

h1 {{
    font-size: 32pt !important;
    line-height: 1.15 !important;
    letter-spacing: -0.02em !important;
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background-color: {COLOR_SURFACE} !important;
    border-right: 1px solid #E2E8F0 !important;
}}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {{
    color: {COLOR_PRIMARY} !important;
}}

/* KPI Metric Cards — Mist Surface with Slate Borders */
div[data-testid="stMetric"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 2px rgba(18, 53, 91, 0.04) !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stMetric"]:hover {{
    border-color: {COLOR_FORECAST} !important;
    box-shadow: 0 4px 8px rgba(70, 130, 220, 0.08) !important;
}}

div[data-testid="stMetricLabel"] p {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: {COLOR_BASELINE} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}}

div[data-testid="stMetricValue"] div {{
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: {COLOR_PRIMARY} !important;
}}

div[data-testid="stMetricDelta"] {{
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}}

/* Tab Navigation */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding-bottom: 2px !important;
}}

.stTabs [data-baseweb="tab"] {{
    height: 42px !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: {COLOR_BASELINE} !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.15s ease !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: {COLOR_FORECAST} !important;
    background-color: {COLOR_SURFACE} !important;
}}

.stTabs [aria-selected="true"] {{
    background-color: {COLOR_SURFACE} !important;
    color: {COLOR_FORECAST} !important;
    border-bottom: 3px solid {COLOR_FORECAST} !important;
}}

/* Top Hero Header Card */
.hero-header {{
    background-color: {COLOR_SURFACE};
    border: 1px solid #E2E8F0;
    border-left: 6px solid {COLOR_PRIMARY};
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
}}

.badge-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.badge-navy {{ background-color: {COLOR_PRIMARY}; color: {COLOR_BG}; }}
.badge-blue {{ background-color: {COLOR_BAND_HEX}; color: {COLOR_FORECAST}; }}
.badge-green {{ background-color: #E8F5E9; color: {COLOR_SAVING}; }}
.badge-amber {{ background-color: #FFF3E0; color: {COLOR_HIGHLIGHT}; }}

/* Big Single Number Class (54pt) */
.big-number-saving {{
    font-size: 54pt;
    font-weight: 800;
    color: {COLOR_SAVING};
    line-height: 1.05;
}}

.big-number-forecast {{
    font-size: 54pt;
    font-weight: 800;
    color: {COLOR_FORECAST};
    line-height: 1.05;
}}

/* Dataframe & Tables Styling */
div[data-testid="stDataFrame"] {{
    background-color: {COLOR_BG};
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
}}
</style>
"""

st.markdown(CUSTOM_BRAND_CSS, unsafe_allow_html=True)


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
# TAB 1: FORECAST (Hero Chart with P10–P90 Band and Peak Markers)
# ==============================================================================

def render_forecast_tab(df: pd.DataFrame, peak_flags: pd.DataFrame, horizon: str) -> None:
    st.subheader(f"Probabilistic Demand Forecast ({horizon.upper()}-Ahead)")

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

    sel_model = c1.selectbox("Forecast Model", models, index=def_idx, key=f"sel_model_{horizon}")
    work = work[work["model"] == sel_model]

    folds = sorted(work["fold"].dropna().unique().tolist())
    sel_fold = c2.selectbox("Walk-Forward Fold (Month)", ["All Folds"] + [str(f) for f in folds], key=f"sel_fold_{horizon}")
    if sel_fold != "All Folds":
        work = work[work["fold"].astype(str) == sel_fold]

    # KPIs
    inside_band = ((work["actual"] >= work["p10"]) & (work["actual"] <= work["p90"])).mean() * 100.0
    rmse = float(np.sqrt(np.mean((work["actual"] - work["p50"]) ** 2)))
    peak_act = float(work["actual"].max())

    k1, k2, k3 = st.columns(3)
    k1.metric("Peak Observed Load", f"{peak_act:,.0f} MW")
    k2.metric("Point Forecast RMSE", f"{rmse:,.1f} MW")
    picp_delta = f"{inside_band - 80.0:+.1f}% vs 80% Target"
    k3.metric("Empirical PICP (P10–P90)", f"{inside_band:.1f}%", picp_delta)

    # Plotly Hero Chart
    fig = go.Figure()

    # P90 upper bound boundary
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p90"], mode="lines",
        line=dict(color=COLOR_FORECAST, width=0),
        name="P90 Upper Bound", showlegend=False
    ))

    # P10 lower bound boundary with Blue Tint shaded band (#4682DC @ 25% opacity)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p10"], mode="lines",
        line=dict(color=COLOR_FORECAST, width=0),
        fill="tonexty",
        fillcolor=COLOR_BAND,
        name="P10–P90 Band (Blue Tint)",
    ))

    # P50 point forecast (Signal Blue dashed line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p50"], mode="lines",
        line=dict(color=COLOR_FORECAST, width=2.4, dash="dash"),
        name="P50 Forecast (Signal Blue)",
    ))

    # Actual demand (Ink #111827 solid line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["actual"], mode="lines",
        line=dict(color=COLOR_ACTUAL, width=2.2),
        name="Actual Demand (Ink)",
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
                    marker=dict(color=COLOR_HIGHLIGHT, size=10, symbol="triangle-up", line=dict(color=COLOR_BG, width=1.5)),
                    name="Predicted Peak Hour (Amber)",
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
                    marker=dict(color=COLOR_PEAK, size=10, symbol="star", line=dict(color=COLOR_BG, width=1.2)),
                    name="Actual Peak Event (Peak Red)",
                ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        colorway=PLOTLY_COLORWAY,
        height=460,
        margin=dict(l=24, r=24, t=36, b=24),
        font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
        xaxis=dict(
            title=dict(text="Time (IST)", font=dict(color=COLOR_PRIMARY, size=12)),
            showgrid=True,
            gridcolor="#EAEFF5",
            linecolor=COLOR_BASELINE,
            tickfont=dict(color=COLOR_PRIMARY, size=11),
        ),
        yaxis=dict(
            title=dict(text="Demand (MW)", font=dict(color=COLOR_PRIMARY, size=12)),
            showgrid=True,
            gridcolor="#EAEFF5",
            linecolor=COLOR_BASELINE,
            tickfont=dict(color=COLOR_PRIMARY, size=11),
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor=COLOR_SURFACE,
            borderwidth=1,
            font=dict(size=11, color=COLOR_PRIMARY),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: ACCURACY VS BASELINES (Strict Model Colors Everywhere)
# ==============================================================================

def render_accuracy_tab(metrics: pd.DataFrame, horizon: str) -> None:
    st.subheader(f"Walk-Forward Evaluation: Model vs Baselines ({horizon.upper()}-Ahead)")
    if metrics.empty:
        st.info("No metrics available.")
        return

    m_sub = metrics[metrics["horizon"] == horizon].copy()
    avg_metrics = m_sub.groupby(["model", "metric"], as_index=False)["value"].mean()
    pivoted = avg_metrics.pivot(index="model", columns="metric", values="value").round(2)

    st.dataframe(pivoted, use_container_width=True)

    # Bar charts comparing MAPE & RMSE using strict brand model colors
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
                textfont=dict(color=COLOR_PRIMARY, size=11),
                marker_line_width=1,
                marker_line_color=COLOR_SURFACE,
            )
            fig_mape.update_layout(
                template="plotly_white",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_BG,
                colorway=PLOTLY_COLORWAY,
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
                xaxis=dict(title="", tickfont=dict(size=11, color=COLOR_PRIMARY)),
                yaxis=dict(title="MAPE (%)", gridcolor="#EAEFF5", tickfont=dict(color=COLOR_PRIMARY)),
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
                textfont=dict(color=COLOR_PRIMARY, size=11),
                marker_line_width=1,
                marker_line_color=COLOR_SURFACE,
            )
            fig_rmse.update_layout(
                template="plotly_white",
                paper_bgcolor=COLOR_BG,
                plot_bgcolor=COLOR_BG,
                colorway=PLOTLY_COLORWAY,
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
                xaxis=dict(title="", tickfont=dict(size=11, color=COLOR_PRIMARY)),
                yaxis=dict(title="RMSE (MW)", gridcolor="#EAEFF5", tickfont=dict(color=COLOR_PRIMARY)),
            )
            st.plotly_chart(fig_rmse, use_container_width=True)


# ==============================================================================
# TAB 3: ERROR HEATMAP (Brand Blue/Navy Gradient)
# ==============================================================================

def render_heatmap_tab(df: pd.DataFrame) -> None:
    st.subheader("Hour × Season Mean Absolute Error Heatmap (MW)")
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

    # Brand sequential gradient: Mist -> Blue Tint -> Signal Blue -> Deep Navy
    brand_gradient = [
        [0.0, COLOR_SURFACE],      # #F3F6FA (Mist)
        [0.35, COLOR_BAND_HEX],     # #D9E6F9 (Blue Tint)
        [0.70, COLOR_FORECAST],     # #4682DC (Signal Blue)
        [1.0, COLOR_PRIMARY],       # #12355B (Deep Navy)
    ]

    fig = px.imshow(
        heat,
        labels=dict(x="Hour of Day (IST)", y="Season", color="MAE (MW)"),
        color_continuous_scale=brand_gradient,
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor=COLOR_SURFACE,
            tickfont=dict(color=COLOR_PRIMARY, size=11),
        ),
        yaxis=dict(
            tickfont=dict(color=COLOR_PRIMARY, size=12, weight="bold"),
        ),
        coloraxis_colorbar=dict(
            title=dict(text="MAE (MW)", font=dict(size=11, color=COLOR_PRIMARY)),
            tickfont=dict(size=10, color=COLOR_PRIMARY),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border-left: 4px solid {COLOR_FORECAST}; padding: 14px 20px; border-radius: 6px; font-size: 13px; color: {COLOR_PRIMARY};">
            <b>Operational Pattern:</b> Error peaks visibly cluster in Summer (MAMJ) evening hours (19:00 – 23:00 IST) when non-linear air-conditioning loads surge during nocturnal heat buildup.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 4: MISSED PEAKS (Peak Red Warnings)
# ==============================================================================

def render_missed_peaks_tab(missed: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Honest Failure Analysis: Missed Peak Events")
    if missed.empty:
        st.success("No peak events were missed by the P90 threshold detector!")
        return

    st.markdown(f"**Days where actual demand exceeded top-decile threshold without P90 peak detection (<span style='color: {COLOR_PEAK};'>Peak Red</span>):**", unsafe_allow_html=True)
    st.dataframe(missed, use_container_width=True, hide_index=True)

    dates = missed["date"].astype(str).tolist()
    sel_date = st.selectbox("Inspect specific missed peak day", dates)
    row = missed[missed["date"].astype(str) == sel_date].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Actual Peak", f"{row['actual_peak_mw']:,.0f} MW")
    k2.metric("Predicted P50", f"{row['pred_p50_peak_mw']:,.0f} MW")
    k3.metric("Predicted P90", f"{row['pred_p90_peak_mw']:,.0f} MW")
    k4.metric("Timing Error", f"{row['hour_error_min']:,.0f} min")

    st.markdown(
        f"""
        <div style="background-color: #FDF2F2; border-left: 4px solid {COLOR_PEAK}; padding: 14px 20px; border-radius: 6px; font-size: 13px; color: #9B1C1C; margin: 16px 0;">
            <b>Diagnostic Note ({sel_date}):</b> {row.get('note', 'Sudden thermal/weather shift')}
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
            line=dict(color=COLOR_PEAK, width=0),
            name="P90 Upper Bound", showlegend=False
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p10"], mode="lines",
            line=dict(color=COLOR_PEAK, width=0),
            fill="tonexty",
            fillcolor="rgba(214, 69, 69, 0.20)",
            name="P10–P90 Missed Range",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p50"], mode="lines",
            line=dict(dash="dash", color=COLOR_FORECAST, width=2.4),
            name="P50 Forecast (Signal Blue)",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["actual"], mode="lines+markers",
            line=dict(color=COLOR_ACTUAL, width=2.2),
            marker=dict(size=6, color=COLOR_PEAK),
            name="Actual Demand (Peak Surge)",
        ))

        fig_day.update_layout(
            template="plotly_white",
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            height=340,
            margin=dict(l=20, r=20, t=24, b=20),
            font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
            yaxis_title="Load (MW)",
            xaxis_title="Time of Day (IST)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_day, use_container_width=True)


# ==============================================================================
# TAB 5: PROCUREMENT COPILOT (Grid Green Savings & Navy/Slate Schedules)
# ==============================================================================

def render_procurement_tab(proc: dict) -> None:
    st.subheader("Procurement Copilot: Weekly Risk-Aware Cost Optimization")
    if not proc:
        st.warning("No procurement results available.")
        return

    c_p90 = proc.get("cost_p90", proc.get("cost_model", 0.0)) / 1e7
    c_flat = proc.get("cost_flat", 0.0) / 1e7
    c_p50 = proc.get("cost_p50", 0.0) / 1e7
    saving = proc.get("saving_p90_vs_flat", proc.get("saving", 0.0)) / 1e7
    pct = proc.get("saving_pct", proc.get("saving_pct_vs_flat", 0.0))
    exposure = proc.get("p90_exposure_mw", 0.0)

    # Hero Rupee Savings Banner (Mist Surface, Grid Green Headline, 54pt number)
    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid #E2E8F0; border-left: 6px solid {COLOR_SAVING}; border-radius: 12px; padding: 24px 30px; margin-bottom: 24px;">
            <div style="font-size: 12px; font-weight: 700; color: {COLOR_SAVING}; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">Summer Peak Procurement Window (Fold 6 — 168 Hours)</div>
            <div class="big-number-saving">₹{saving:,.2f} Cr</div>
            <div style="font-size: 18px; font-weight: 600; color: {COLOR_PRIMARY}; margin-top: 4px;">
                Net Rupee Savings (+{pct:.2f}% Cost Reduction vs Naive Flat Schedule)
            </div>
            <div style="font-size: 13px; color: {COLOR_BASELINE}; margin-top: 8px;">
                Dynamic P90 scheduling prevents unhedged exposure to punitive ₹12.00/kWh DSM penalties while avoiding excess stranded capacity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P90 Dynamic Schedule Cost", f"₹{c_p90:,.2f} Cr")
    k2.metric("P50 Median Schedule Cost", f"₹{c_p50:,.2f} Cr")
    k3.metric("Naive Flat Schedule Cost", f"₹{c_flat:,.2f} Cr")
    k4.metric("Max P90 Risk Exposure", f"{exposure:,.1f} MW")

    st.markdown(
        f"""
        <div style="background-color: {COLOR_SURFACE}; border: 1px solid #E2E8F0; padding: 12px 18px; border-radius: 8px; font-size: 12px; color: {COLOR_PRIMARY}; margin: 16px 0;">
            <b>Tariff Assumptions:</b> Base Contracted Rate = ₹5.00/kWh | DSM Shortfall Penalty = ₹12.00/kWh | Stranded Surplus Loss = ₹2.50/kWh
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hourly detail chart
    if "hourly_detail_p90" in proc and len(proc["hourly_detail_p90"]) > 0:
        h_df = pd.DataFrame(proc["hourly_detail_p90"])
        h_df["ts"] = pd.to_datetime(h_df["ts"])

        st.markdown("**Hourly Schedule vs Actual Demand During Peak Summer Week (168 Hours)**")
        fig_sched = go.Figure()

        # Flat naive schedule (Slate dotted line)
        if "schedule_flat" in proc:
            flat_val = float(proc["schedule_flat"][0]) if isinstance(proc["schedule_flat"], list) else float(proc["schedule_flat"])
            fig_sched.add_trace(go.Scatter(
                x=h_df["ts"], y=[flat_val] * len(h_df), mode="lines",
                line=dict(color=COLOR_BASELINE, width=2, dash="dot"),
                name="Naive Flat Schedule (Slate)",
            ))

        # Dynamic P90 Schedule (Grid Green line)
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["schedule_mw"], mode="lines",
            line=dict(color=COLOR_SAVING, width=2.4),
            name="Dynamic P90 Schedule (Grid Green)",
        ))

        # Actual Demand (Ink solid line)
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["actual_mw"], mode="lines",
            line=dict(color=COLOR_ACTUAL, width=2.2),
            name="Actual Demand (Ink)",
        ))

        fig_sched.update_layout(
            template="plotly_white",
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Calibri, Inter, sans-serif", color=COLOR_PRIMARY),
            yaxis=dict(
                title=dict(text="Load (MW)", font=dict(color=COLOR_PRIMARY, size=12)),
                gridcolor="#EAEFF5",
                linecolor=COLOR_BASELINE,
                tickfont=dict(color=COLOR_PRIMARY, size=11),
            ),
            xaxis=dict(
                title=dict(text="Time (IST)", font=dict(color=COLOR_PRIMARY, size=12)),
                gridcolor="#EAEFF5",
                linecolor=COLOR_BASELINE,
                tickfont=dict(color=COLOR_PRIMARY, size=11),
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor=COLOR_SURFACE,
                borderwidth=1,
                font=dict(size=11, color=COLOR_PRIMARY),
            ),
        )
        st.plotly_chart(fig_sched, use_container_width=True)

    # Sensitivity Table
    if "sensitivity_table" in proc:
        st.markdown("**Tariff Sensitivity Analysis (DSM vs Stranded Waste Rates)**")
        sens_df = pd.DataFrame(proc["sensitivity_table"])
        st.dataframe(sens_df, use_container_width=True, hide_index=True)


# ==============================================================================
# MAIN APP ENTRYPOINT
# ==============================================================================

def main():
    # Hero Top Header
    st.markdown(
        f"""
        <div class="hero-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span class="badge-tag badge-navy">Delhi DISCOM</span>
                        <span class="badge-tag badge-blue">● Signal Blue Copilot</span>
                        <span class="badge-tag badge-green">Grid Green Savings</span>
                    </div>
                    <div style="font-size: 26px; font-weight: 800; color: {COLOR_PRIMARY}; margin: 0; font-family: Calibri, Inter, sans-serif;">
                        Probabilistic Peak-Aware Load Forecasting & Procurement Copilot
                    </div>
                    <div style="font-size: 14px; color: {COLOR_BASELINE}; margin-top: 4px;">
                        Hourly P10/P50/P90 Quantile Trajectories • Walk-Forward Calibration • Risk-Optimized Power Procurement
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
        st.markdown(f"### 📋 **Contract Interfaces**")
        for f in ["forecasts_day.csv", "forecasts_week.csv", "metrics.csv", "peak_flags.csv", "missed_peaks.csv", "procurement_week.json"]:
            p = RESULTS_DIR / f
            if p.exists():
                st.markdown(f"<div style='font-size: 12px; margin-bottom: 4px;'><span style='color: {COLOR_SAVING}; font-weight: bold;'>✓</span> <code>{f}</code></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size: 12px; margin-bottom: 4px;'><span style='color: {COLOR_PEAK}; font-weight: bold;'>✗</span> <code>{f}</code></div>", unsafe_allow_html=True)

        st.divider()
        st.caption("🔒 Offline dashboard reading `results/` only.")

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
