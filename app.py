"""
app.py — Delhi DISCOM Load Forecasting & Procurement Copilot Dashboard
======================================================================
Reads results/ files ONLY. Never imports models/, features/, or data/.
Runs 100% offline.

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

st.set_page_config(
    page_title="Delhi DISCOM Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"


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
# TAB 1: FORECAST (Hero Chart with P10-P90 Band and Peak Markers)
# ==============================================================================

def render_forecast_tab(df: pd.DataFrame, peak_flags: pd.DataFrame, horizon: str) -> None:
    st.subheader(f"Probabilistic Demand Forecast ({horizon.upper()}-Ahead) with P10–P90 Range")

    if df.empty:
        st.warning("No forecast data available.")
        return

    work = df.copy().sort_values("ts")

    c1, c2 = st.columns(2)
    models = sorted(work["model"].dropna().unique().tolist())
    def_idx = 0
    for i, m in enumerate(models):
        if "lgbm" in m:
            def_idx = i
            break
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
    k3.metric("Empirical PICP (Target 80%)", f"{inside_band:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=work["ts"], y=work["p90"], mode="lines", line=dict(width=0), name="P90", showlegend=False))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p10"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(37, 99, 235, 0.22)", name="P10–P90 Uncertainty Band",
    ))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p50"], mode="lines",
        line=dict(color="#1d4ed8", width=2, dash="dash"), name="P50 Expected Load",
    ))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["actual"], mode="lines",
        line=dict(color="#111827", width=2), name="Actual Demand (MW)",
    ))

    # Peak markers
    if not peak_flags.empty:
        pf = peak_flags.copy()
        pf["date_str"] = pd.to_datetime(pf["date"]).dt.strftime("%Y-%m-%d")
        work["date_str"] = work["ts"].dt.strftime("%Y-%m-%d")
        work["hour"] = work["ts"].dt.hour

        pred_peaks = pf[pf["peak_pred"].astype(bool)]
        if not pred_peaks.empty:
            merged = work.merge(pred_peaks[["date_str", "peak_hour_pred", "peak_prob"]],
                                left_on=["date_str", "hour"], right_on=["date_str", "peak_hour_pred"], how="inner")
            if not merged.empty:
                fig.add_trace(go.Scatter(
                    x=merged["ts"], y=merged["p90"], mode="markers",
                    marker=dict(color="#f59e0b", size=10, symbol="triangle-up"),
                    name="Predicted Peak Hour",
                ))

        actual_peaks = pf[pf["peak_actual"].astype(bool)]
        if not actual_peaks.empty:
            merged_act = work.merge(actual_peaks[["date_str", "peak_hour_actual"]],
                                    left_on=["date_str", "hour"], right_on=["date_str", "peak_hour_actual"], how="inner")
            if not merged_act.empty:
                fig.add_trace(go.Scatter(
                    x=merged_act["ts"], y=merged_act["actual"], mode="markers",
                    marker=dict(color="#dc2626", size=9, symbol="star"),
                    name="Actual Peak Event (>90th percentile)",
                ))

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (IST)",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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

    # Bar chart comparing MAPE & RMSE
    piv_reset = pivoted.reset_index()
    if "MAPE" in piv_reset.columns and "RMSE" in piv_reset.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig_mape = px.bar(piv_reset, x="model", y="MAPE", color="model", title="MAPE (%) — Lower is better")
            fig_mape.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_mape, use_container_width=True)
        with c2:
            fig_rmse = px.bar(piv_reset, x="model", y="RMSE", color="model", title="RMSE (MW) — Lower is better")
            fig_rmse.update_layout(showlegend=False, height=320)
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

    fig = px.imshow(
        heat,
        labels=dict(x="Hour of Day (IST)", y="Season", color="MAE (MW)"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Error spikes occur during summer evening hours (19:00 - 23:00 IST) due to cooling demand swings.")


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

    st.info(f"**Failure Analysis Note ({sel_date})**: {row.get('note', 'Sudden thermal/weather shift')}")

    # Zoom in on that day
    f_sub = df.copy()
    f_sub["date_str"] = f_sub["ts"].dt.strftime("%Y-%m-%d")
    day_slice = f_sub[f_sub["date_str"] == sel_date].sort_values("ts")
    if not day_slice.empty:
        pref = [m for m in day_slice["model"].unique() if "lgbm" in m]
        if pref:
            day_slice = day_slice[day_slice["model"] == pref[0]]
        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(x=day_slice["ts"], y=day_slice["p90"], mode="lines", line=dict(width=0), name="P90"))
        fig_day.add_trace(go.Scatter(
            x=day_slice["ts"], y=day_slice["p10"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239, 68, 68, 0.2)", name="P10–P90 Range"
        ))
        fig_day.add_trace(go.Scatter(x=day_slice["ts"], y=day_slice["p50"], mode="lines", line=dict(dash="dash", color="#2563eb"), name="P50"))
        fig_day.add_trace(go.Scatter(x=day_slice["ts"], y=day_slice["actual"], mode="lines+markers", line=dict(color="#dc2626"), name="Actual Demand"))
        fig_day.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Load (MW)")
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
    saving = proc.get("saving_p90_vs_flat", proc.get("saving", 0.0)) / 1e7
    pct = proc.get("saving_pct", proc.get("saving_pct_vs_flat", 0.0))
    exposure = proc.get("p90_exposure_mw", 0.0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P90 Dynamic Schedule Cost", f"Rs {c_p90:,.2f} Cr")
    k2.metric("Naive Flat Schedule Cost", f"Rs {c_flat:,.2f} Cr")
    k3.metric("Simulated Weekly Savings", f"Rs {saving:,.2f} Cr", f"{pct:.2f}% vs Flat")
    k4.metric("Max P90 Risk Exposure", f"{exposure:,.1f} MW")

    st.info(
        "**Marked Tariff Assumptions**: Contracted = Rs 5.00/kWh | "
        "DSM Penalty Rate = Rs 12.00/kWh | Stranded Surplus Loss = Rs 2.50/kWh"
    )

    # Hourly procurement breakdown or sensitivity
    if "sensitivity_table" in proc:
        st.markdown("**Tariff Sensitivity Analysis (DSM vs Stranded Rates)**")
        sens_df = pd.DataFrame(proc["sensitivity_table"])
        st.dataframe(sens_df, use_container_width=True, hide_index=True)


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.title("Delhi DISCOM: Probabilistic Peak-Aware Forecasting & Procurement Copilot")
    st.caption("CodeSpark 2026 | Offline Operational Dashboard reading results/ only")

    with st.sidebar:
        st.header("Settings")
        horizon = st.radio("Forecast Horizon", ["day", "week"], index=0, format_func=lambda h: "Day-Ahead (24h)" if h == "day" else "Week-Ahead (168h)")
        st.divider()
        st.markdown("**Data Contracts Status**")
        for f in ["forecasts_day.csv", "forecasts_week.csv", "metrics.csv", "peak_flags.csv", "missed_peaks.csv", "procurement_week.json"]:
            p = RESULTS_DIR / f
            status = "FOUND" if p.exists() else "MISSING"
            st.caption(f"• `{f}`: {status}")

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
