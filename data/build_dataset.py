"""
Canonical Dataset Builder for Delhi DISCOM Load & Weather (Data + Dashboard / P2).

Contract #1: data/clean_hourly.parquet
    ts                   : datetime64[ns, Asia/Kolkata], hourly, unique, sorted
    load_mw              : float64 (Megawatts)
    load_mw_imputed_flag : bool (True where value was filled or replaced from outage)
    temp_c               : float64
    rh_pct               : float64
    apparent_temp_c      : float64
    is_holiday           : bool
    festival_name        : str
    is_oddeven           : bool
    is_lockdown          : bool

Outputs:
    - data/clean_hourly.parquet
    - data/imputation_log.md

Note: Does NOT compute model features, baselines, backtests, metrics, or procurement.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IST_TZ = "Asia/Kolkata"
DELHI_LAT = 28.6139
DELHI_LON = 77.2090

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

EXPECTED_COLUMNS = [
    "ts",
    "load_mw",
    "load_mw_imputed_flag",
    "temp_c",
    "rh_pct",
    "apparent_temp_c",
    "is_holiday",
    "festival_name",
    "is_oddeven",
    "is_lockdown",
]


@dataclass
class ImputationEntry:
    column: str
    count: int
    method: str
    reason: str
    details: str = ""


@dataclass
class ImputationTracker:
    source_load: str = ""
    source_weather: str = ""
    is_fallback_load: bool = False
    is_fallback_weather: bool = False
    raw_rows_load: int = 0
    raw_rows_weather: int = 0
    final_rows: int = 0
    start_ts: str = ""
    end_ts: str = ""
    entries: list[ImputationEntry] = field(default_factory=list)
    gap_summary: list[dict[str, Any]] = field(default_factory=list)

    def add(self, column: str, count: int, method: str, reason: str, details: str = "") -> None:
        self.entries.append(
            ImputationEntry(column=column, count=int(count), method=method, reason=reason, details=details)
        )


# ---------------------------------------------------------------------------
# 1. Delhi Calendar (Holidays, Major Festivals, Odd-Even, Lockdowns)
# ---------------------------------------------------------------------------

# Major fixed and moving festivals in Delhi (2019-2026)
MOVING_FESTIVALS: dict[str, str] = {
    # 2019
    "2019-03-04": "Maha Shivaratri",
    "2019-03-21": "Holi",
    "2019-06-05": "Eid-ul-Fitr",
    "2019-08-12": "Eid-ul-Adha",
    "2019-08-15": "Raksha Bandhan / Independence Day",
    "2019-08-24": "Janmashtami",
    "2019-10-08": "Dussehra",
    "2019-10-27": "Diwali",
    "2019-10-28": "Govardhan Puja",
    "2019-11-12": "Guru Nanak Jayanti",
    # 2020
    "2020-02-21": "Maha Shivaratri",
    "2020-03-10": "Holi",
    "2020-05-25": "Eid-ul-Fitr",
    "2020-08-01": "Eid-ul-Adha",
    "2020-08-03": "Raksha Bandhan",
    "2020-08-12": "Janmashtami",
    "2020-10-25": "Dussehra",
    "2020-11-14": "Diwali",
    "2020-11-15": "Govardhan Puja",
    "2020-11-30": "Guru Nanak Jayanti",
    # 2021
    "2021-03-11": "Maha Shivaratri",
    "2021-03-29": "Holi",
    "2021-05-14": "Eid-ul-Fitr",
    "2021-07-21": "Eid-ul-Adha",
    "2021-08-22": "Raksha Bandhan",
    "2021-08-30": "Janmashtami",
    "2021-10-15": "Dussehra",
    "2021-11-04": "Diwali",
    "2021-11-05": "Govardhan Puja",
    "2021-11-19": "Guru Nanak Jayanti",
    # 2022
    "2022-03-01": "Maha Shivaratri",
    "2022-03-18": "Holi",
    "2022-05-03": "Eid-ul-Fitr",
    "2022-07-10": "Eid-ul-Adha",
    "2022-08-11": "Raksha Bandhan",
    "2022-08-19": "Janmashtami",
    "2022-10-05": "Dussehra",
    "2022-10-24": "Diwali",
    "2022-10-25": "Govardhan Puja",
    "2022-11-08": "Guru Nanak Jayanti",
    # 2023
    "2023-02-18": "Maha Shivaratri",
    "2023-03-08": "Holi",
    "2023-04-22": "Eid-ul-Fitr",
    "2023-06-29": "Eid-ul-Adha",
    "2023-08-30": "Raksha Bandhan",
    "2023-09-07": "Janmashtami",
    "2023-10-24": "Dussehra",
    "2023-11-12": "Diwali",
    "2023-11-13": "Govardhan Puja",
    "2023-11-27": "Guru Nanak Jayanti",
    # 2024
    "2024-03-08": "Maha Shivaratri",
    "2024-03-25": "Holi",
    "2024-04-11": "Eid-ul-Fitr",
    "2024-06-17": "Eid-ul-Adha",
    "2024-08-19": "Raksha Bandhan",
    "2024-08-26": "Janmashtami",
    "2024-10-12": "Dussehra",
    "2024-10-31": "Diwali",
    "2024-11-01": "Diwali (Delhi Govt) / Govardhan Puja",
    "2024-11-15": "Guru Nanak Jayanti",
    # 2025
    "2025-02-26": "Maha Shivaratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Eid-ul-Fitr",
    "2025-06-07": "Eid-ul-Adha",
    "2025-08-09": "Raksha Bandhan",
    "2025-08-16": "Janmashtami",
    "2025-10-02": "Dussehra / Gandhi Jayanti",
    "2025-10-20": "Diwali",
    "2025-11-05": "Guru Nanak Jayanti",
    # 2026
    "2026-02-15": "Maha Shivaratri",
    "2026-03-04": "Holi",
    "2026-03-20": "Eid-ul-Fitr",
    "2026-05-27": "Eid-ul-Adha",
    "2026-10-20": "Dussehra",
    "2026-11-08": "Diwali",
    "2026-11-24": "Guru Nanak Jayanti",
}

FIXED_HOLIDAYS: dict[tuple[int, int], str] = {
    (1, 26): "Republic Day",
    (4, 14): "Ambedkar Jayanti",
    (8, 15): "Independence Day",
    (10, 2): "Gandhi Jayanti",
    (12, 25): "Christmas",
}

# Delhi lockdown windows (2020 national/state & 2021 Delta wave)
LOCKDOWN_WINDOWS: list[tuple[str, str]] = [
    ("2020-03-25", "2020-05-31"),
    ("2021-04-19", "2021-05-31"),
]

# Delhi Odd-Even traffic scheme periods
ODDEVEN_WINDOWS: list[tuple[str, str]] = [
    ("2019-11-04", "2019-11-15"),
]


def build_delhi_calendar(ts_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Construct Delhi holiday, festival, odd-even, and lockdown columns aligned to ts_index."""
    date_strs = ts_index.strftime("%Y-%m-%d")
    months = ts_index.month
    days = ts_index.day

    festival_names: list[str] = []
    is_holiday: list[bool] = []
    for d_str, m, d in zip(date_strs, months, days):
        fest = MOVING_FESTIVALS.get(d_str, "")
        fixed = FIXED_HOLIDAYS.get((int(m), int(d)), "")
        if fest and fixed and fixed not in fest:
            combined = f"{fest} / {fixed}"
        else:
            combined = fest or fixed
        festival_names.append(combined)
        is_holiday.append(bool(combined))

    dates_pd = pd.to_datetime(date_strs)
    is_lockdown = np.zeros(len(ts_index), dtype=bool)
    for start_d, end_d in LOCKDOWN_WINDOWS:
        is_lockdown |= (dates_pd >= start_d) & (dates_pd <= end_d)

    is_oddeven = np.zeros(len(ts_index), dtype=bool)
    for start_d, end_d in ODDEVEN_WINDOWS:
        is_oddeven |= (dates_pd >= start_d) & (dates_pd <= end_d)

    return pd.DataFrame({
        "is_holiday": pd.Series(is_holiday, dtype=bool),
        "festival_name": pd.Series(festival_names, dtype=str),
        "is_oddeven": pd.Series(is_oddeven, dtype=bool),
        "is_lockdown": pd.Series(is_lockdown, dtype=bool),
    })


# ---------------------------------------------------------------------------
# 2. Data Ingestion (Real CSV / Open-Meteo API / Clearly-Labelled Fallback)
# ---------------------------------------------------------------------------

def _to_ist_hourly(series: pd.Series) -> pd.Series:
    """Parse timestamps and localize/convert to Asia/Kolkata rounded to hour start."""
    dt = pd.to_datetime(series, dayfirst=False, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(IST_TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        dt = dt.dt.tz_convert(IST_TZ)
    return dt.dt.floor("h")


def _infer_column(columns: list[str], candidates: tuple[str, ...], override: str | None) -> str:
    if override:
        if override not in columns:
            raise ValueError(f"Column '{override}' not found in {columns}")
        return override
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"Could not infer column from candidates {candidates} in {columns}")


def load_raw_load_csv(
    csv_path: Path,
    ts_col: str | None = None,
    load_col: str | None = None,
) -> pd.DataFrame:
    """Load real hourly Delhi load CSV (SLDC / Grid-India / Kaggle) into ['ts', 'load_mw']."""
    raw = pd.read_csv(csv_path)
    cols = list(raw.columns)
    t_col = _infer_column(cols, ("ts", "timestamp", "datetime", "date_time", "date", "time"), ts_col)
    l_col = _infer_column(cols, ("load_mw", "load", "delhi", "demand_mw", "demand", "power_mw"), load_col)

    out = pd.DataFrame({
        "ts": _to_ist_hourly(raw[t_col]),
        "load_mw": pd.to_numeric(raw[l_col], errors="coerce"),
    })
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    assert out["ts"].is_monotonic_increasing, "Loaded timestamps must be monotonically increasing"
    return out


def fetch_open_meteo_weather(start_date: str, end_date: str, timeout_s: int = 20) -> pd.DataFrame:
    """
    Fetch historical hourly weather for Delhi from Open-Meteo Archive API (free, no key).
    Variables: temperature_2m, relative_humidity_2m, apparent_temperature.
    """
    params = urllib.parse.urlencode({
        "latitude": DELHI_LAT,
        "longitude": DELHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature",
        "timezone": IST_TZ,
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Delhi-DISCOM-Hackathon/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(f"Unexpected Open-Meteo response keys: {list(payload.keys())}")

    return pd.DataFrame({
        "ts": _to_ist_hourly(pd.Series(hourly["time"])),
        "temp_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
        "rh_pct": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
        "apparent_temp_c": pd.to_numeric(hourly["apparent_temperature"], errors="coerce"),
    })


def generate_fallback_weather(start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    """
    CLEARLY LABELLED FALLBACK: Synthetic Delhi hourly weather when offline / Open-Meteo unreachable.
    Models seasonal cycle (winter ~12 C to summer ~42 C), monsoon RH surge, and diurnal swings.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{start_date} 00:00:00", f"{end_date} 23:00:00", freq="h", tz=IST_TZ)
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    n = len(ts)

    # Seasonal baseline temperature in Delhi (peak May-June, trough Jan)
    seasonal_mean_c = 25.0 + 11.5 * np.sin(2.0 * np.pi * (doy - 100.0) / 365.25)
    # Diurnal cycle: coolest ~05:30, warmest ~15:00
    diurnal_c = 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    # Multi-day synoptic weather wave (heatwaves / western disturbances)
    synoptic = pd.Series(rng.normal(0.0, 0.45, n)).rolling(72, min_periods=1).sum().to_numpy() * 0.35
    temp_c = np.clip(seasonal_mean_c + diurnal_c + synoptic + rng.normal(0.0, 0.6, n), 3.0, 48.0)

    # Monsoon humidity spike July-Sep + inverse diurnal relation with temperature
    monsoon_rh = 22.0 * np.exp(-0.5 * ((doy - 220.0) / 35.0) ** 2)
    winter_fog_rh = 15.0 * np.exp(-0.5 * (((doy % 365) - 15.0) / 25.0) ** 2)
    rh_pct = np.clip(50.0 + monsoon_rh + winter_fog_rh - 1.4 * diurnal_c + rng.normal(0.0, 4.0, n), 12.0, 98.0)

    # Apparent temperature approximation (Steadman-like humidex boost above 26 C)
    heat_boost = np.maximum(0.0, temp_c - 26.0) * (rh_pct / 100.0) * 0.45
    apparent_temp_c = temp_c + heat_boost - 1.0

    return pd.DataFrame({
        "ts": ts,
        "temp_c": np.round(temp_c, 2),
        "rh_pct": np.round(rh_pct, 2),
        "apparent_temp_c": np.round(apparent_temp_c, 2),
    })


def generate_fallback_delhi_load(weather_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    CLEARLY LABELLED FALLBACK: Synthetic hourly Delhi load (~2,000 MW winter to ~8,300 MW summer)
    with realistic morning ramp (6-10 AM), evening peak (7-11 PM), cooling load above 25 C,
    72h thermal inertia, festival/lockdown impacts, AND injected realistic gaps/duplicates/outages
    so the cleaning pipeline is faithfully exercised.
    """
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(weather_df["ts"])
    n = len(ts)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    years_elapsed = (ts - ts[0]).total_seconds().to_numpy() / (365.25 * 24 * 3600)

    temp = weather_df["temp_c"].to_numpy(dtype=float)
    rh = weather_df["rh_pct"].to_numpy(dtype=float)
    temp_72h = pd.Series(temp).rolling(72, min_periods=1).mean().to_numpy()

    # Non-linear cooling degree + thermal inertia + winter morning heating bump
    cdh = np.maximum(0.0, temp - 25.0)
    inertia_cdh = np.maximum(0.0, temp_72h - 27.0)
    winter_heat = np.maximum(0.0, 15.0 - temp) * 40.0

    # Diurnal shape: trough ~04:30, morning ramp 06-10, afternoon plateau, evening peak 19-23
    morning_ramp = 650.0 * np.exp(-0.5 * ((hour - 9.5) / 2.2) ** 2)
    afternoon_ac = 750.0 * np.exp(-0.5 * ((hour - 15.0) / 3.0) ** 2) * (cdh > 0)
    evening_peak = 1250.0 * np.exp(-0.5 * ((hour - 22.0) / 2.3) ** 2)
    night_trough = -600.0 * np.exp(-0.5 * ((hour - 4.5) / 2.0) ** 2)

    base_mw = 2550.0 + 125.0 * years_elapsed  # ~3-4% annual load growth
    cooling_mw = 135.0 * cdh + 1.8 * (cdh ** 2) + 85.0 * inertia_cdh + 0.35 * cdh * np.maximum(0.0, rh - 45.0)

    cal = build_delhi_calendar(ts)
    weekend_adj = np.where(dow >= 5, -240.0, 0.0)
    holiday_adj = np.where(cal["is_holiday"].to_numpy(), -320.0, 0.0)
    lockdown_adj = np.where(cal["is_lockdown"].to_numpy(), -1150.0, 0.0)

    load_mw = (
        base_mw
        + cooling_mw
        + winter_heat
        + morning_ramp
        + afternoon_ac
        + evening_peak
        + night_trough
        + weekend_adj
        + holiday_adj
        + lockdown_adj
        + rng.normal(0.0, 95.0, size=n)
    )
    load_mw = np.clip(load_mw, 1850.0, 8450.0)

    df = pd.DataFrame({"ts": ts, "load_mw": np.round(load_mw, 2)})

    # Inject small realistic telemetry anomalies (short gap, 10h gap, 2 outage zeros, 2 duplicates)
    if n > 500:
        df.loc[120:122, "load_mw"] = np.nan           # 3h short meter gap
        df.loc[340:349, "load_mw"] = np.nan           # 10h long outage gap
        df.loc[410, "load_mw"] = 0.0                  # outage zero spike
        df.loc[460, "load_mw"] = 14900.0              # sensor upper spike
        dup_rows = df.iloc[[200, 201]].copy()
        df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# 3. Cleaning, Outlier Flagging & Gap Imputation
# ---------------------------------------------------------------------------

def _run_lengths_of_nans(is_na: np.ndarray) -> np.ndarray:
    """Return length of contiguous NaN run at each index (0 where not NaN)."""
    n = len(is_na)
    run_len = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if not is_na[i]:
            i += 1
            continue
        j = i
        while j < n and is_na[j]:
            j += 1
        run_len[i:j] = j - i
        i = j
    return run_len


def clean_and_impute_load(
    raw_load: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    tracker: ImputationTracker,
) -> tuple[pd.Series, pd.Series]:
    """
    Deduplicate, reindex to full IST hourly grid, flag outages/spikes, and impute gaps.
    Short gaps (<= 6h): linear interpolation.
    Longer gaps (> 6h): same-hour-last-week (t - 168h), then forward/seasonal median fallback.
    """
    df = raw_load.copy()
    dup_count = int(df.duplicated(subset=["ts"]).sum())
    if dup_count > 0:
        df = df.groupby("ts", as_index=False)["load_mw"].mean()
    tracker.add(
        "load_mw",
        dup_count,
        "Mean aggregation by ts",
        "Duplicate hourly timestamps in raw feed",
    )

    s = df.set_index("ts")["load_mw"].reindex(full_index).astype(float)
    missing_raw_count = int(s.isna().sum())
    tracker.add(
        "load_mw",
        missing_raw_count,
        "Reindexed to complete IST hourly grid",
        "Missing timestamps or raw NaN readings before outlier check",
    )

    # Outlier / outage detection (flag and set NaN, never silently delete rows)
    roll_med_7d = s.rolling(168, min_periods=24, center=True).median()
    roll_med_24h = s.rolling(24, min_periods=6, center=True).median()
    low_outage = (s < 500.0) | (roll_med_7d.notna() & (s < 0.20 * roll_med_7d))
    high_spike = (s > 11000.0) | (roll_med_24h.notna() & ((s - roll_med_24h).abs() > 3000.0))
    outlier_mask = (low_outage | high_spike).fillna(False)

    outlier_count = int(outlier_mask.sum())
    if outlier_count > 0:
        sample_ts = [str(t) for t in full_index[outlier_mask][:5]]
        s[outlier_mask] = np.nan
        tracker.add(
            "load_mw",
            outlier_count,
            "Flagged -> NaN -> imputed",
            "Outage (<500 MW or <20% 7d median) or telemetry spike (>11,000 MW / >3,000 MW jump)",
            details=f"Sample flagged timestamps: {', '.join(sample_ts)}",
        )
    else:
        tracker.add("load_mw", 0, "Range & rolling-median check", "No outage/spike outliers detected")

    # Track gap lengths before filling
    na_mask = s.isna().to_numpy()
    imputed_flag = pd.Series(na_mask, index=full_index, dtype=bool)
    run_lens = _run_lengths_of_nans(na_mask)

    short_mask = na_mask & (run_lens <= 6)
    long_mask = na_mask & (run_lens > 6)

    # Step 1: Linear interpolation ONLY for short gaps (<= 6h)
    s_interp = s.interpolate(method="linear", limit=6, limit_area="inside")
    s_filled = s.copy()
    s_filled[short_mask] = s_interp[short_mask]
    short_filled = int(short_mask.sum() - s_filled[short_mask].isna().sum())
    tracker.add("load_mw", short_filled, "Linear interpolation (<=6h)", "Short telemetry/clock gaps")

    # Step 2: Same-hour-last-week (t - 168h) for longer gaps (> 6h) or edge NaNs
    arr = s_filled.to_numpy(dtype=float)
    weekly_filled = 0
    for idx in range(len(arr)):
        if np.isnan(arr[idx]) and idx >= 168 and not np.isnan(arr[idx - 168]):
            arr[idx] = arr[idx - 168]
            weekly_filled += 1
    tracker.add(
        "load_mw",
        weekly_filled,
        "Same-hour-last-week (t - 168h)",
        "Longer outages (>6h) or leading/trailing short gaps",
    )

    # Step 3: Fallback for any remaining NaNs at the very start of the dataset
    s_final = pd.Series(arr, index=full_index, dtype=float)
    remaining = int(s_final.isna().sum())
    if remaining > 0:
        # Try same-hour-next-week backward pass, then hour-of-day median
        for idx in range(len(arr) - 169, -1, -1):
            if np.isnan(arr[idx]) and not np.isnan(arr[idx + 168]):
                arr[idx] = arr[idx + 168]
        s_final = pd.Series(arr, index=full_index, dtype=float)
        still_na = s_final.isna()
        if still_na.any():
            hod_median = s_final.groupby(s_final.index.hour).transform("median")
            s_final = s_final.fillna(hod_median).fillna(float(s_final.median()))
        tracker.add(
            "load_mw",
            remaining,
            "Same-hour-next-week / hour-of-day median",
            "Leading-edge gaps lacking prior-week history",
        )

    # Record gap distribution summary
    unique_long_runs = int(np.sum(long_mask))
    tracker.gap_summary.append({
        "total_imputed_hours": int(imputed_flag.sum()),
        "short_gap_hours_le_6h": int(short_mask.sum()),
        "long_gap_hours_gt_6h": unique_long_runs,
        "outliers_flagged": outlier_count,
    })

    return s_final.round(2), imputed_flag


def clean_and_impute_weather(
    raw_weather: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    tracker: ImputationTracker,
) -> pd.DataFrame:
    """Align weather to full IST hourly index and impute any gaps cleanly."""
    df = raw_weather.copy()
    dup_count = int(df.duplicated(subset=["ts"]).sum())
    if dup_count > 0:
        df = df.groupby("ts", as_index=False)[["temp_c", "rh_pct", "apparent_temp_c"]].mean()
        tracker.add("weather_*", dup_count, "Mean aggregation by ts", "Duplicate weather timestamps")

    aligned = df.set_index("ts").reindex(full_index)
    for col in ("temp_c", "rh_pct", "apparent_temp_c"):
        s = pd.to_numeric(aligned[col], errors="coerce").astype(float)
        n_miss = int(s.isna().sum())
        if n_miss > 0:
            s = s.interpolate(method="linear", limit=6, limit_area="inside")
            if s.isna().any():
                hod_med = s.groupby(s.index.hour).transform("median")
                s = s.fillna(hod_med).fillna(float(s.median()))
            tracker.add(col, n_miss, "Linear interp (<=6h) + diurnal median", "Missing weather hours")
        else:
            tracker.add(col, 0, "None needed", "Complete hourly weather coverage")
        aligned[col] = s.round(2)

    return aligned.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Schema Validation Block (Contract #1)
# ---------------------------------------------------------------------------

def validate_clean_hourly(df: pd.DataFrame) -> None:
    """Assert every Contract #1 requirement on data/clean_hourly.parquet."""
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Column mismatch! Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )
    assert len(df) > 0, "clean_hourly DataFrame is empty."

    ts = df["ts"]
    assert isinstance(ts.dtype, pd.DatetimeTZDtype), f"ts must be tz-aware DatetimeTZDtype, got {ts.dtype}"
    assert str(ts.dt.tz) == IST_TZ, f"ts timezone must be {IST_TZ}, got {ts.dt.tz}"
    assert ts.is_unique, "ts contains duplicate timestamps!"
    assert ts.is_monotonic_increasing, "ts is not sorted ascending!"

    diffs = ts.diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all(), "ts has missing hours or non-hourly spacing!"

    for col in ("load_mw", "temp_c", "rh_pct", "apparent_temp_c"):
        assert pd.api.types.is_float_dtype(df[col]), f"{col} must be float dtype, got {df[col].dtype}"
        assert df[col].notna().all(), f"{col} contains NaNs!"

    for col in ("load_mw_imputed_flag", "is_holiday", "is_oddeven", "is_lockdown"):
        assert pd.api.types.is_bool_dtype(df[col]), f"{col} must be bool dtype, got {df[col].dtype}"
        assert df[col].notna().all(), f"{col} contains NaNs!"

    assert df["festival_name"].notna().all(), "festival_name contains NaNs!"
    assert df["load_mw"].min() > 0, f"Unexpected non-positive load_mw: {df['load_mw'].min()}"


# ---------------------------------------------------------------------------
# 5. Imputation Log Generator (data/imputation_log.md)
# ---------------------------------------------------------------------------

def write_imputation_log(tracker: ImputationTracker, out_path: Path) -> None:
    """Write human-readable imputation and data-quality audit log to data/imputation_log.md."""
    lines = [
        "# Data Cleaning & Imputation Log (`data/clean_hourly.parquet`)",
        "",
        f"- **Hourly Window (IST)**: `{tracker.start_ts}` to `{tracker.end_ts}` (`{tracker.final_rows:,}` hours)",
        f"- **Load Source**: {tracker.source_load}",
        f"- **Weather Source**: {tracker.source_weather}",
    ]
    if tracker.is_fallback_load or tracker.is_fallback_weather:
        lines.extend([
            "",
            "> **[FALLBACK NOTICE — FOR MODEL CARD]**: One or more inputs used the clearly-labelled synthetic fallback generator because no local raw CSV / live Open-Meteo connection was supplied. Replace via `--load-csv` and `--fetch-open-meteo` when real files are placed locally.",
        ])

    lines.extend([
        "",
        "## Summary of Load Gap & Outlier Handling",
    ])
    for gs in tracker.gap_summary:
        lines.extend([
            f"- Total imputed load hours (`load_mw_imputed_flag == True`): **{gs['total_imputed_hours']:,}** ({100.0 * gs['total_imputed_hours'] / max(1, tracker.final_rows):.3f}%)",
            f"- Short gaps (<= 6h, linear interpolation): **{gs['short_gap_hours_le_6h']:,}** hours",
            f"- Longer gaps (> 6h, same-hour-last-week `t - 168h`): **{gs['long_gap_hours_gt_6h']:,}** hours",
            f"- Outage / telemetry spike outliers flagged & re-imputed: **{gs['outliers_flagged']:,}** hours",
        ])

    lines.extend([
        "",
        "## Detailed Per-Column Operations",
        "",
        "| Column | Count | Method | Reason | Notes |",
        "| --- | ---: | --- | --- | --- |",
    ])
    for entry in tracker.entries:
        lines.append(
            f"| `{entry.column}` | {entry.count:,} | {entry.method} | {entry.reason} | {entry.details or '—'} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Pipeline Entrypoint
# ---------------------------------------------------------------------------

def build_dataset(
    load_csv: Path | None = None,
    weather_csv: Path | None = None,
    fetch_open_meteo: bool = False,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    out_parquet: Path = DATA_DIR / "clean_hourly.parquet",
    out_log: Path = DATA_DIR / "imputation_log.md",
    ts_col: str | None = None,
    load_col: str | None = None,
    allow_csv_if_no_parquet_engine: bool = False,
) -> pd.DataFrame:
    """Run full Phase 1 (H0-H3) data pipeline and assert Contract #1 schema."""
    tracker = ImputationTracker()

    # Step 1: Load or generate weather first so fallback load can reference weather if needed
    if weather_csv is not None and weather_csv.exists():
        raw_w = pd.read_csv(weather_csv)
        raw_w["ts"] = _to_ist_hourly(raw_w[_infer_column(list(raw_w.columns), ("ts", "timestamp", "time", "datetime"), None)])
        raw_weather = raw_w[["ts", "temp_c", "rh_pct", "apparent_temp_c"]].copy()
        tracker.source_weather = f"Local CSV ({weather_csv.name})"
    elif fetch_open_meteo:
        try:
            raw_weather = fetch_open_meteo_weather(start_date, end_date)
            tracker.source_weather = f"Open-Meteo Historical Archive API (lat={DELHI_LAT}, lon={DELHI_LON}, IST)"
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"[WARN] Open-Meteo fetch failed ({exc}); using clearly-labelled synthetic fallback weather.")
            raw_weather = generate_fallback_weather(start_date, end_date)
            tracker.source_weather = "FALLBACK: Synthetic Delhi hourly weather (API unreachable)"
            tracker.is_fallback_weather = True
    else:
        raw_weather = generate_fallback_weather(start_date, end_date)
        tracker.source_weather = "FALLBACK: Synthetic Delhi hourly weather (offline default)"
        tracker.is_fallback_weather = True
    tracker.raw_rows_weather = len(raw_weather)

    # Step 2: Load or generate Delhi hourly load
    if load_csv is not None and load_csv.exists():
        raw_load = load_raw_load_csv(load_csv, ts_col=ts_col, load_col=load_col)
        tracker.source_load = f"Local CSV ({load_csv.name})"
    else:
        raw_load = generate_fallback_delhi_load(raw_weather)
        tracker.source_load = "FALLBACK: Synthetic Delhi hourly load (2,000-8,300 MW diurnal + CDH + thermal inertia)"
        tracker.is_fallback_load = True
    tracker.raw_rows_load = len(raw_load)

    # Step 3: Build canonical IST hourly index spanning load range
    min_ts = raw_load["ts"].min()
    max_ts = raw_load["ts"].max()
    full_index = pd.date_range(start=min_ts, end=max_ts, freq="h", tz=IST_TZ)

    # If real load CSV extended beyond fallback weather, regenerate weather on matching dates
    if tracker.is_fallback_weather and (raw_weather["ts"].min() > min_ts or raw_weather["ts"].max() < max_ts):
        raw_weather = generate_fallback_weather(min_ts.strftime("%Y-%m-%d"), max_ts.strftime("%Y-%m-%d"))

    # Step 4: Clean & impute load and weather
    load_series, imputed_flag = clean_and_impute_load(raw_load, full_index, tracker)
    weather_clean = clean_and_impute_weather(raw_weather, full_index, tracker)
    calendar_df = build_delhi_calendar(full_index)

    clean_df = pd.DataFrame({
        "ts": full_index,
        "load_mw": load_series.to_numpy(dtype=float),
        "load_mw_imputed_flag": imputed_flag.to_numpy(dtype=bool),
        "temp_c": weather_clean["temp_c"].to_numpy(dtype=float),
        "rh_pct": weather_clean["rh_pct"].to_numpy(dtype=float),
        "apparent_temp_c": weather_clean["apparent_temp_c"].to_numpy(dtype=float),
        "is_holiday": calendar_df["is_holiday"].to_numpy(dtype=bool),
        "festival_name": calendar_df["festival_name"].astype(str).to_numpy(),
        "is_oddeven": calendar_df["is_oddeven"].to_numpy(dtype=bool),
        "is_lockdown": calendar_df["is_lockdown"].to_numpy(dtype=bool),
    })

    tracker.final_rows = len(clean_df)
    tracker.start_ts = str(full_index[0])
    tracker.end_ts = str(full_index[-1])

    # Step 5: Assert Contract #1 schema invariants
    validate_clean_hourly(clean_df)
    print(f"[OK] Schema validation passed: {len(clean_df):,} hourly IST rows ({tracker.start_ts} -> {tracker.end_ts})")

    # Step 6: Write outputs
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        clean_df.to_parquet(out_parquet, index=False)
        print(f"[OK] Wrote {out_parquet}")
    except ImportError as exc:
        if not allow_csv_if_no_parquet_engine:
            raise RuntimeError(
                "Writing data/clean_hourly.parquet requires pyarrow (pip install pyarrow==26.0.1)."
            ) from exc
        fallback_csv = out_parquet.with_suffix(".csv")
        clean_df.to_csv(fallback_csv, index=False)
        print(f"[WARN] pyarrow not installed; wrote verification CSV -> {fallback_csv}")

    write_imputation_log(tracker, out_log)
    print(f"[OK] Wrote {out_log}")
    return clean_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Delhi DISCOM hourly dataset.")
    parser.add_argument("--load-csv", type=Path, default=None, help="Path to raw hourly Delhi load CSV.")
    parser.add_argument("--weather-csv", type=Path, default=None, help="Path to raw hourly weather CSV.")
    parser.add_argument("--fetch-open-meteo", action="store_true", help="Fetch hourly weather from Open-Meteo API.")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date YYYY-MM-DD.")
    parser.add_argument("--ts-col", type=str, default=None, help="Timestamp column name in --load-csv.")
    parser.add_argument("--load-col", type=str, default=None, help="Load MW column name in --load-csv.")
    parser.add_argument(
        "--allow-csv-if-no-parquet-engine",
        action="store_true",
        help="Dry-run fallback if pyarrow is not yet installed in the environment.",
    )
    args = parser.parse_args()

    build_dataset(
        load_csv=args.load_csv,
        weather_csv=args.weather_csv,
        fetch_open_meteo=args.fetch_open_meteo,
        start_date=args.start,
        end_date=args.end,
        ts_col=args.ts_col,
        load_col=args.load_col,
        allow_csv_if_no_parquet_engine=args.allow_csv_if_no_parquet_engine,
    )


if __name__ == "__main__":
    main()

