"""
Inspection script for data/raw/powerdemand_5min_2021_to_2024_with weather.csv
"""

import os
import pandas as pd
import numpy as np

raw_path = os.path.join("data", "raw", "powerdemand_5min_2021_to_2024_with weather.csv")
print("File exists:", os.path.exists(raw_path))

df = pd.read_csv(raw_path)
print(f"Row count: {len(df):,}")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

start_date = df["datetime"].min()
end_date = df["datetime"].max()
print(f"Date range: {start_date} to {end_date}")

# Demand and Temp ranges
demand_min = df["Power demand"].min()
demand_max = df["Power demand"].max()
temp_min = df["temp"].min()
temp_max = df["temp"].max()
print(f"Power demand: min = {demand_min:.2f} MW, max = {demand_max:.2f} MW")
print(f"Temperature:  min = {temp_min:.2f} C, max = {temp_max:.2f} C")

# Gaps
full_range = pd.date_range(start=start_date, end=end_date, freq="5min")
print(f"Total expected 5-min intervals: {len(full_range):,}")
print(f"Total missing timestamps: {len(full_range) - len(df):,}")

df["gap"] = df["datetime"].diff()
longest = df.nlargest(5, "gap")
print("\n5 Longest gaps:")
for _, r in longest.iterrows():
    p_time = r["datetime"] - r["gap"]
    print(f"  From {p_time} to {r['datetime']}: {r['gap']}")

# Missing per month
df["ym"] = df["datetime"].dt.to_period("M")
act_cnt = df.groupby("ym").size()

ym_range = pd.period_range(start_date.to_period("M"), end_date.to_period("M"), freq="M")
print("\nMissing 5-min intervals per month:")
june_aug = []
for p in ym_range:
    m_start = p.start_time
    # End of month
    m_end = p.end_time
    expected = len(pd.date_range(m_start, m_end, freq="5min"))
    actual = act_cnt.get(p, 0)
    diff = expected - actual
    if p.month in (6, 7, 8):
        june_aug.append((str(p), expected, actual, diff))
    if diff > 0:
        print(f"  {p}: {diff:,} missing (expected {expected:,}, actual {actual:,})")

print("\nJune-August Coverage (Summer Peaks):")
for p_str, exp, act, diff in june_aug:
    pct = (act / exp) * 100 if exp > 0 else 0
    print(f"  {p_str}: {act:,} / {exp:,} rows ({pct:.1f}% present, {diff:,} missing)")

