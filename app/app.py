"""
Delhi DISCOM Probabilistic Peak-Aware Load & Procurement Copilot (Data + Dashboard / P2).

Contract #6:
    - Reads results/ files ONLY (via app/dummy_loader.py).
    - NEVER imports models/, features/, eval/, procurement/, or data/.
    - Runs 100% offline.

Required Views:
    1. Hero chart: actual vs P50 with shaded P10-P90 band and peak markers.
    2. Accuracy table per horizon vs baselines + Hour x Season error heatmap.
    3. Missed-peak drill-down: pick a day from missed_peaks.csv, show forecast vs actual & note.
    4. Procurement panel from results/procurement_week.json.

Launch:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.dummy_loader import load_dashboard_bundle

st.set_page_config(
    page_title="Delhi DISCOM Probabilistic Load & Procurement Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter (DJF)"
    if month in (3, 4, 5, 6):
        return "Summer (MAMJ)"
    if month in (7, 8, 9):
        return "Monsoon (JAS)"
    return "Post-Monsoon (ON)"


def render_hero_chart(df: pd.DataFrame, peak_flags: pd.DataFrame, horizon: str) -> None:
    """View 1: Actual vs P50 with shaded P10-P90 band and peak markers."""
    st.subheader(f"1. Probabilistic Load Forecast ({horizon.title()}-Ahead) & Peak Markers")

    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work = work.sort_values("ts")

    # Controls for model and fold
    c1, c2 = st.columns(2)
    models = sorted(work["model"].dropna().unique().tolist()) if "model" in work.columns else ["model"]
    default_idx = 0
    for idx, m in enumerate(models):
        if "naive" not in m.lower() and "linear" not in m.lower():
            default_idx = idx
            break
    sel_model = c1.selectbox("Forecast Model", models, index=default_idx, key=f"hero_model_{horizon}")
    if "model" in work.columns:
        work = work[work["model"] == sel_model]

    folds = sorted(work["fold"].dropna().unique().tolist()) if "fold" in work.columns else ["all"]
    sel_fold = c2.selectbox("Walk-Forward Fold (Month)", ["All Folds"] + [str(f) for f in folds], key=f"hero_fold_{horizon}")
    if sel_fold != "All Folds" and "fold" in work.columns:
        work = work[work["fold"].astype(str) == sel_fold]

    # Summary KPIs for displayed slice
    inside_band = ((work["actual"] >= work["p10"]) & (work["actual"] <= work["p90"])).mean() * 100.0
    rmse = float(np.sqrt(np.mean((work["actual"] - work["p50"]) ** 2)))
    peak_actual = float(work["actual"].max())

    k1, k2, k3 = st.columns(3)
    k1.metric("Peak Actual Load", f"{peak_actual:,.0f} MW")
    k2.metric("Slice RMSE (P50)", f"{rmse:,.1f} MW")
    k3.metric("Empirical PICP (Target ~80%)", f"{inside_band:.1f}%")

    fig = go.Figure()
    # Upper & lower P10-P90 shaded band
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p90"],
        mode="lines", line=dict(width=0),
        name="P90", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p10"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(59, 130, 246, 0.22)",
        name="P10–P90 Band",
    ))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["p50"],
        mode="lines", line=dict(color="#1d4ed8", width=2, dash="dash"),
        name="P50 Forecast",
    ))
    fig.add_trace(go.Scatter(
        x=work["ts"], y=work["actual"],
        mode="lines", line=dict(color="#111827", width=2),
        name="Actual Load (MW)",
    ))

    # Overlay peak-event markers from peak_flags.parquet
    if not peak_flags.empty:
        pf = peak_flags.copy()
        pf["date_str"] = pd.to_datetime(pf["date"]).dt.strftime("%Y-%m-%d")
        work["date_str"] = work["ts"].dt.strftime("%Y-%m-%d")
        work["hour"] = work["ts"].dt.hour

        pred_peaks = pf[pf["peak_pred"].astype(bool)]
        if not pred_peaks.empty:
            merged_pred = work.merge(
                pred_peaks[["date_str", "peak_hour_pred", "peak_prob"]],
                left_on=["date_str", "hour"],
                right_on=["date_str", "peak_hour_pred"],
                how="inner",
            )
            if not merged_pred.empty:
                fig.add_trace(go.Scatter(
                    x=merged_pred["ts"], y=merged_pred["p90"],
                    mode="markers",
                    marker=dict(color="#f59e0b", size=10, symbol="triangle-up"),
                    name="Predicted Peak Hour",
                    text=[f"Peak Prob: {p:.2f}" for p in merged_pred["peak_prob"]],
                ))

        actual_peaks = pf[pf["peak_actual"].astype(bool)]
        if not actual_peaks.empty:
            merged_act = work.merge(
                actual_peaks[["date_str", "peak_hour_actual"]],
                left_on=["date_str", "hour"],
                right_on=["date_str", "peak_hour_actual"],
                how="inner",
            )
            if not merged_act.empty:
                fig.add_trace(go.Scatter(
                    x=merged_act["ts"], y=merged_act["actual"],
                    mode="markers",
                    marker=dict(color="#dc2626", size=9, symbol="star"),
                    name="Actual Top-Decile Peak",
                ))

    fig.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Timestamp (IST)",
        yaxis_title="Load (MW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_evaluation_section(metrics: pd.DataFrame, forecasts_df: pd.DataFrame, horizon: str) -> None:
    """View 2: Accuracy table vs baselines + Hour x Season error heatmap."""
    st.subheader(f"2. Walk-Forward Evaluation & Hour × Season Error Heatmap ({horizon.title()}-Ahead)")

    col_tbl, col_heat = st.columns([1.05, 1.0])

    with col_tbl:
        st.markdown("**Per-Horizon Metrics vs. Baselines (`results/metrics.csv`)**")
        m_sub = metrics[metrics["horizon"] == horizon].copy()
        if "fold" in m_sub.columns and (m_sub["fold"].astype(str) == "all").any():
            m_sub = m_sub[m_sub["fold"].astype(str) == "all"]
        elif not m_sub.empty:
            m_sub = m_sub.groupby(["model", "metric"], as_index=False)["value"].mean()

        if not m_sub.empty:
            pivoted = m_sub.pivot(index="model", columns="metric", values="value").round(3)
            st.dataframe(pivoted, use_container_width=True)
        else:
            st.info(f"No rows for horizon='{horizon}' in metrics.csv.")

    with col_heat:
        st.markdown("**Hour × Season Mean Absolute Error (MW)**")
        f_sub = forecasts_df.copy()
        if "model" in f_sub.columns:
            preferred = [m for m in f_sub["model"].unique() if "naive" not in m.lower() and "linear" not in m.lower()]
            if preferred:
                f_sub = f_sub[f_sub["model"] == preferred[0]]

        f_sub["ts"] = pd.to_datetime(f_sub["ts"])
        f_sub["hour"] = f_sub["ts"].dt.hour
        f_sub["season"] = f_sub["ts"].dt.month.map(_season_from_month)
        f_sub["abs_err_mw"] = (f_sub["actual"] - f_sub["p50"]).abs()

        season_order = ["Winter (DJF)", "Summer (MAMJ)", "Monsoon (JAS)", "Post-Monsoon (ON)"]
        heat = (
            f_sub.pivot_table(index="season", columns="hour", values="abs_err_mw", aggfunc="mean")
            .reindex([s for s in season_order if s in f_sub["season"].unique()])
        )
        fig_heat = px.imshow(
            heat,
            labels=dict(x="Hour of Day (IST)", y="Season", color="MAE (MW)"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig_heat.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)


def render_missed_peaks_drilldown(missed_peaks: pd.DataFrame, forecasts_df: pd.DataFrame) -> None:
    """View 3: Missed-peak drill-down from results/missed_peaks.csv."""
    st.subheader("3. Honest Failure Analysis: Missed-Peak Day Drill-Down")
    if missed_peaks.empty:
        st.info("No missed peaks recorded in results/missed_peaks.csv.")
        return

    st.dataframe(missed_peaks, use_container_width=True, hide_index=True)

    dates = missed_peaks["date"].astype(str).tolist()
    chosen_date = st.selectbox("Select a Missed-Peak Date to Inspect", dates, key="missed_peak_date")
    row = missed_peaks[missed_peaks["date"].astype(str) == chosen_date].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Peak", f"{row['actual_peak_mw']:,.0f} MW")
    c2.metric("Pred P50 Peak", f"{row['pred_p50_peak_mw']:,.0f} MW")
    c3.metric("Pred P90 Peak", f"{row['pred_p90_peak_mw']:,.0f} MW")
    c4.metric("Peak Timing Error", f"{row['hour_error_min']:,.0f} min")
    st.warning(f"**Diagnosed Failure Reason ({chosen_date})**: {row['note']}")

    # Plot 24-hour zoom if matching date exists in forecasts
    f_sub = forecasts_df.copy()
    f_sub["ts"] = pd.to_datetime(f_sub["ts"])
    f_sub["date_str"] = f_sub["ts"].dt.strftime("%Y-%m-%d")
    day_df = f_sub[f_sub["date_str"] == chosen_date].sort_values("ts")
    if not day_df.empty:
        if "model" in day_df.columns:
            preferred = [m for m in day_df["model"].unique() if "naive" not in m.lower() and "linear" not in m.lower()]
            if preferred:
                day_df = day_df[day_df["model"] == preferred[0]]
        fig_day = go.Figure()
        fig_day.add_trace(go.Scatter(x=day_df["ts"], y=day_df["p90"], mode="lines", line=dict(width=0), showlegend=False))
        fig_day.add_trace(go.Scatter(
            x=day_df["ts"], y=day_df["p10"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239, 68, 68, 0.18)", name="P10–P90 Band",
        ))
        fig_day.add_trace(go.Scatter(x=day_df["ts"], y=day_df["p50"], mode="lines", line=dict(color="#2563eb", dash="dash"), name="P50"))
        fig_day.add_trace(go.Scatter(x=day_df["ts"], y=day_df["actual"], mode="lines+markers", line=dict(color="#dc2626"), name="Actual"))
        fig_day.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Load (MW)")
        st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.caption("Hourly profile for this specific date is outside the currently loaded sample slice.")


def render_procurement_panel(proc: dict) -> None:
    """View 4: Procurement Copilot panel from results/procurement_week.json."""
    st.subheader("4. Procurement Copilot: P90 Buy Schedule & Rupee Savings")

    cost_model_cr = proc["cost_model"] / 1e7
    cost_flat_cr = proc["cost_flat"] / 1e7
    saving_cr = proc["saving"] / 1e7
    saving_pct = proc.get("saving_pct_vs_flat", (proc["saving"] / proc["cost_flat"] * 100.0) if proc["cost_flat"] else 0.0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("P90 Model Cost", f"Rs {cost_model_cr:,.2f} Cr")
    k2.metric("Naive Flat Schedule Cost", f"Rs {cost_flat_cr:,.2f} Cr")
    k3.metric("Simulated Weekly Saving", f"Rs {saving_cr:,.2f} Cr", f"{saving_pct:.2f}% vs Flat")
    k4.metric("P90 Tail Exposure", f"{proc['p90_exposure_mw']:,.1f} MW")

    assump = proc.get("assumptions", {})
    st.info(
        f"**Marked Tariff Assumptions**: Contracted = Rs {assump.get('contracted_rate_rs_per_kwh', 5.0)}/kWh | "
        f"DSM Shortfall = Rs {assump.get('dsm_rate_rs_per_kwh', 12.0)}/kWh | "
        f"Stranded Surplus = Rs {assump.get('stranded_rate_rs_per_kwh', 2.5)}/kWh"
    )

    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        sched_24x7 = np.array(proc["schedule"], dtype=float)
        days = proc.get("schedule_days", [f"Day {i+1}" for i in range(sched_24x7.shape[1])])
        fig_sched = px.imshow(
            sched_24x7,
            x=days,
            y=list(range(sched_24x7.shape[0])),
            labels=dict(x="Day", y="Hour (IST)", color="Buy (MW)"),
            color_continuous_scale="Blues",
            aspect="auto",
            title="24×7 P90 Day-Ahead Buy Schedule (MW)",
        )
        fig_sched.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sched, use_container_width=True)

    with col_right:
        breakdown = proc.get("cost_breakdown")
        if breakdown:
            rows = []
            for label, key in [("P90 Schedule", "p90_model"), ("P50 Schedule", "p50_point"), ("Flat (mean P90)", "flat_p90_avg")]:
                if key in breakdown:
                    b = breakdown[key]
                    rows.append({"Strategy": label, "Component": "Contracted", "Rs (Crore)": b["contracted_cost_rs"] / 1e7})
                    rows.append({"Strategy": label, "Component": "DSM Shortfall Penalty", "Rs (Crore)": b["dsm_shortfall_cost_rs"] / 1e7})
                    rows.append({"Strategy": label, "Component": "Stranded Surplus", "Rs (Crore)": b["stranded_surplus_cost_rs"] / 1e7})
            fig_bar = px.bar(
                pd.DataFrame(rows),
                x="Strategy",
                y="Rs (Crore)",
                color="Component",
                barmode="stack",
                title="Weekly Procurement Cost Breakdown",
            )
            fig_bar.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    if "sensitivity" in proc and proc["sensitivity"]:
        with st.expander("Tariff Sensitivity Analysis (DSM & Stranded Rates)"):
            sens_df = pd.DataFrame(proc["sensitivity"])
            st.dataframe(sens_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Delhi DISCOM: Probabilistic Peak-Aware Load & Procurement Copilot")
    st.caption("CodeSpark 2026 | Offline Dashboard reading saved results/ files only")

    bundle = load_dashboard_bundle()
    status = bundle["sources_status"]
    any_dummy = any("DUMMY" in v for v in status.values())

    with st.sidebar:
        st.header("Controls & Data Status")
        horizon = st.radio("Forecast Horizon", ["day", "week"], index=1, format_func=lambda h: "Day-Ahead (24h)" if h == "day" else "Week-Ahead (168h)")
        st.divider()
        st.markdown("**Loaded Result Sources**")
        for k, v in status.items():
            st.caption(f"- `{k}`: {v}")

    if any_dummy:
        st.warning(
            "**Pre-H6 Demo Mode**: One or more `results/` files are currently using dummy/stub data. "
            "As soon as Features & Modeling write real files to `results/`, refresh to view real backtest results."
        )

    active_forecasts = bundle["forecasts_day"] if horizon == "day" else bundle["forecasts_week"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Hero Forecast & Peaks",
        "Evaluation & Heatmap",
        "Missed-Peak Drill-Down",
        "Procurement Copilot",
    ])

    with tab1:
        render_hero_chart(active_forecasts, bundle["peak_flags"], horizon)
    with tab2:
        render_evaluation_section(bundle["metrics"], active_forecasts, horizon)
    with tab3:
        render_missed_peaks_drilldown(bundle["missed_peaks"], active_forecasts)
    with tab4:
        render_procurement_panel(bundle["procurement"])


if __name__ == "__main__":
    main()

