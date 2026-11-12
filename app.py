"""
app.py — Delhi DISCOM Load Forecasting & Procurement Copilot Dashboard
======================================================================
Reads results/ files ONLY. Never imports models/, features/, or data/.
Runs 100% offline.

Theme: Modern Light UI/UX with soft corporate palette and accessible contrast.
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
# LIGHT THEME CUSTOM CSS
# ==============================================================================

CUSTOM_LIGHT_CSS = """
<style>
/* Main app background & font settings */
.stApp {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* KPI Metric Cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    border-color: #CBD5E1 !important;
}

div[data-testid="stMetricLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}

div[data-testid="stMetricValue"] div {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* Modern Tab Navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: transparent !important;
    border-bottom: 2px solid #E2E8F0 !important;
    padding-bottom: 2px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 44px !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    background-color: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #2563EB !important;
    background-color: #F1F5F9 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #EFF6FF !important;
    color: #2563EB !important;
    border-bottom: 3px solid #2563EB !important;
}

/* Card containers */
.light-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%);
    border: 1px solid #DBEAFE;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.04);
}

.badge-tag {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-blue { background-color: #DBEAFE; color: #1D4ED8; }
.badge-green { background-color: #DCFCE7; color: #15803D; }
.badge-amber { background-color: #FEF3C7; color: #B45309; }

/* Dataframe & Tables */
div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
}
</style>
"""

st.markdown(CUSTOM_LIGHT_CSS, unsafe_allow_html=True)


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
    st.subheader(f"Probabilistic Demand Forecast ({horizon.upper()}-Ahead) with P10–P90 Range")

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

    # Plotly Light Hero Chart
    fig = go.Figure()

    # P90 upper bound line (invisible, sets upper bound for tonexty)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p90"], mode="lines",
        line=dict(width=0), name="P90 Upper Bound", showlegend=False
    ))

    # P10 lower bound line with soft sky-blue fill
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p10"], mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.16)",
        name="P10–P90 Uncertainty Band (80%)",
    ))

    # P50 point forecast (royal blue dashed line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p50"], mode="lines",
        line=dict(color="#2563EB", width=2, dash="dash"),
        name="P50 Expected Demand",
    ))

    # Actual demand (crisp dark slate line)
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["actual"], mode="lines",
        line=dict(color="#0F172A", width=2.2),
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
                    marker=dict(color="#D97706", size=10, symbol="triangle-up", line=dict(color="#FFFFFF", width=1.5)),
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
                    marker=dict(color="#DC2626", size=9, symbol="star", line=dict(color="#FFFFFF", width=1)),
                    name="Actual Peak Event (>90th %ile)",
                ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=460,
        margin=dict(l=24, r=24, t=36, b=24),
        xaxis=dict(
            title="Time (IST)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(color="#475569", size=12),
            tickfont=dict(color="#64748B", size=11),
        ),
        yaxis=dict(
            title="Demand (MW)",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
            titlefont=dict(color="#475569", size=12),
            tickfont=dict(color="#64748B", size=11),
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11, color="#334155"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: ACCURACY VS BASELINES
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

    # Bar charts comparing MAPE & RMSE in clean light colors
    piv_reset = pivoted.reset_index()
    if "MAPE" in piv_reset.columns and "RMSE" in piv_reset.columns:
        c1, c2 = st.columns(2)

        color_map = {
            "lgbm_quantile": "#2563EB",            # Royal blue
            "lgbm_quantile_calibrated": "#059669", # Emerald green
            "linear_regression": "#6366F1",        # Indigo
            "seasonal_naive": "#94A3B8",           # Cool slate
        }

        with c1:
            fig_mape = px.bar(
                piv_reset, x="model", y="MAPE",
                color="model",
                color_discrete_map=color_map,
                title="<b>MAPE (%)</b> — Lower is better",
                text="MAPE",
            )
            fig_mape.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_mape.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
                yaxis=dict(title="MAPE (%)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
            )
            st.plotly_chart(fig_mape, use_container_width=True)

        with c2:
            fig_rmse = px.bar(
                piv_reset, x="model", y="RMSE",
                color="model",
                color_discrete_map=color_map,
                title="<b>RMSE (MW)</b> — Lower is better",
                text="RMSE",
            )
            fig_rmse.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_rmse.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                showlegend=False,
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(title="", tickfont=dict(size=11, color="#334155")),
                yaxis=dict(title="RMSE (MW)", gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
            )
            st.plotly_chart(fig_rmse, use_container_width=True)


# ==============================================================================
# TAB 3: ERROR HEATMAP (Hour x Season MAE)
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

    # Clean light pastel color scale for heatmap
    fig = px.imshow(
        heat,
        labels=dict(x="Hour of Day (IST)", y="Season", color="MAE (MW)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor="#F8FAFC",
            tickfont=dict(color="#475569", size=11),
        ),
        yaxis=dict(
            tickfont=dict(color="#1E293B", size=12, weight="bold"),
        ),
        coloraxis_colorbar=dict(
            title="MAE (MW)",
            titlefont=dict(size=11, color="#475569"),
            tickfont=dict(size=10, color="#64748B"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style="background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #92400E;">
            <b>Operational Insight:</b> Error spikes visibly concentrate in Summer (MAMJ) evening hours (19:00 – 23:00 IST), driven by non-linear residential air-conditioning saturation and nocturnal heat trapping.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 4: MISSED PEAKS
# ==============================================================================

def render_missed_peaks_tab(missed: pd.DataFrame, df: pd.DataFrame) -> None:
    st.subheader("Honest Failure Analysis: Missed Peak Events")
    if missed.empty:
        st.success("No peak events were missed by the P90 threshold detector!")
        return

    st.markdown("**Days where actual demand exceeded top-decile threshold without P90 peak detection:**")
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
        <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #1E40AF; margin: 16px 0;">
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
            line=dict(width=0), name="P90 Upper Bound", showlegend=False
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p10"], mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.16)",
            name="P10–P90 Range (Missed Band)",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p50"], mode="lines",
            line=dict(dash="dash", color="#2563EB", width=2),
            name="P50 Expected Demand",
        ))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["actual"], mode="lines+markers",
            line=dict(color="#DC2626", width=2.5),
            marker=dict(size=5, color="#DC2626"),
            name="Actual Demand (Peak Surge)",
        ))

        fig_day.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=340,
            margin=dict(l=20, r=20, t=24, b=20),
            yaxis_title="Load (MW)",
            xaxis_title="Time of Day (IST)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_day, use_container_width=True)


# ==============================================================================
# TAB 5: PROCUREMENT COPILOT
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

    # Hero Rupee Savings Banner (Soft Green Light Theme)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%); border: 1px solid #A7F3D0; border-radius: 12px; padding: 20px 24px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(16, 185, 129, 0.08);">
            <div style="font-size: 12px; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Representative Summer Peak Week Savings (Fold 6)</div>
            <div style="font-size: 32px; font-weight: 800; color: #065F46; line-height: 1.2;">
                ₹{saving:,.2f} Crore 
                <span style="font-size: 18px; font-weight: 600; color: #059669; margin-left: 8px;">(+{pct:.2f}% Cost Reduction vs Naive Flat)</span>
            </div>
            <div style="font-size: 13px; color: #047857; margin-top: 6px;">
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
        """
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 16px; border-radius: 8px; font-size: 12px; color: #64748B; margin: 16px 0;">
            <b>Marked Tariff Assumptions:</b> Base Contracted Rate = ₹5.00/kWh | DSM Shortfall Penalty = ₹12.00/kWh | Stranded Surplus Loss = ₹2.50/kWh
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hourly detail chart if present
    if "hourly_detail_p90" in proc and len(proc["hourly_detail_p90"]) > 0:
        h_df = pd.DataFrame(proc["hourly_detail_p90"])
        h_df["ts"] = pd.to_datetime(h_df["ts"])

        st.markdown("**Hourly Schedule vs Actual Demand During Peak Summer Week (168 Hours)**")
        fig_sched = go.Figure()

        # Flat naive schedule
        if "schedule_flat" in proc:
            flat_val = float(proc["schedule_flat"][0]) if isinstance(proc["schedule_flat"], list) else float(proc["schedule_flat"])
            fig_sched.add_trace(go.Scatter(
                x=h_df["ts"], y=[flat_val] * len(h_df), mode="lines",
                line=dict(color="#94A3B8", width=1.8, dash="dot"),
                name="Naive Flat Schedule (Mean P90)",
            ))

        # Dynamic P90 Schedule
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["schedule_mw"], mode="lines",
            line=dict(color="#2563EB", width=2.2),
            name="Dynamic P90 Schedule (Contracted)",
        ))

        # Actual Demand
        fig_sched.add_trace(go.Scatter(
            x=h_df["ts"], y=h_df["actual_mw"], mode="lines",
            line=dict(color="#0F172A", width=2),
            name="Actual Demand (MW)",
        ))

        fig_sched.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Load (MW)",
            xaxis_title="Time (IST)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
        """
        <div class="hero-header">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span class="badge-tag badge-blue">Delhi DISCOM</span>
                        <span class="badge-tag badge-green">● Operational Copilot</span>
                        <span class="badge-tag badge-amber">CodeSpark 2026</span>
                    </div>
                    <div style="font-size: 24px; font-weight: 800; color: #0F172A; margin: 0;">
                        Probabilistic Peak-Aware Load Forecasting & Procurement Copilot
                    </div>
                    <div style="font-size: 13px; color: #64748B; margin-top: 4px;">
                        Hourly P10/P50/P90 Quantile Trajectories • Walk-Forward Calibration • Risk-Optimized Power Procurement
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ **Control Deck**")
        horizon = st.radio(
            "Forecast Horizon",
            ["day", "week"],
            index=0,
            format_func=lambda h: "Day-Ahead (24h)" if h == "day" else "Week-Ahead (168h)"
        )

        st.divider()
        st.markdown("### 📋 **Contract Interfaces**")
        for f in ["forecasts_day.csv", "forecasts_week.csv", "metrics.csv", "peak_flags.csv", "missed_peaks.csv", "procurement_week.json"]:
            p = RESULTS_DIR / f
            if p.exists():
                st.markdown(f"<div style='font-size: 12px; margin-bottom: 4px;'><span style='color: #10B981; font-weight: bold;'>✓</span> <code>{f}</code></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size: 12px; margin-bottom: 4px;'><span style='color: #EF4444; font-weight: bold;'>✗</span> <code>{f}</code></div>", unsafe_allow_html=True)

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
