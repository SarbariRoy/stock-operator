"""
Priority 4 validation: compare st_score vs signal_score as predictors
of 7-day 3%/3% outcomes on closed signals.
"""
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

TARGET_PCT = 3.0
STOP_PCT = 2.0
MAX_DAYS = 7

signals = pd.read_csv("stock_triggers/data/st_signals_all_patterns.csv", parse_dates=["signal_date"])
prices = pd.read_csv("stock_triggers/data/st_lt_prices_eod.csv", parse_dates=["Date"])

print(f"Signals: {len(signals)}, st_score present: {'st_score' in signals.columns}")


def simulate_outcome(row, prices_df):
    ticker = str(row["ticker"])
    sig_date = row["signal_date"]
    entry = float(row["entry_price"])
    if entry <= 0:
        return "no_data"
    target = entry * (1 + TARGET_PCT / 100)
    stop = entry * (1 - STOP_PCT / 100)
    future = (
        prices_df[(prices_df["Ticker"] == ticker) & (prices_df["Date"] > sig_date)]
        .sort_values("Date")
        .head(MAX_DAYS)
    )
    for _, bar in future.iterrows():
        if float(bar["High"]) >= target:
            return "target"
        if float(bar["Close"]) <= stop:
            return "stop"
    return "holding"


print(f"Simulating {len(signals)} signals (7-day, {TARGET_PCT}%/{STOP_PCT}%) …")
signals["outcome"] = signals.apply(lambda r: simulate_outcome(r, prices), axis=1)
print("Done.\n")

closed = signals[signals["outcome"].isin(["target", "stop"])].copy()
closed["hit"] = (closed["outcome"] == "target").astype(int)

n_target = (closed["outcome"] == "target").sum()
n_stop = (closed["outcome"] == "stop").sum()
n_holding = (signals["outcome"] == "holding").sum()
baseline_wr = n_target / len(closed) * 100 if len(closed) else 0
print(f"Overall closed: {len(closed)} | Target: {n_target} | Stop: {n_stop} | Holding: {n_holding}")
print(f"Baseline win rate (closed): {baseline_wr:.1f}%")

# Spearman correlation (manual, no scipy needed)
def spearman_r(x, y):
    n = len(x)
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n ** 2 - 1))

r_st  = spearman_r(closed["st_score"].fillna(0).values,  closed["hit"].values)
r_sig = spearman_r(closed["signal_score"].fillna(0).values, closed["hit"].values)
print(f"\nSpearman correlation with outcome (target=1, stop=0):")
print(f"  st_score:     r={r_st:.4f}")
print(f"  signal_score: r={r_sig:.4f}")

# Win rate by bucket
bins = [0, 30, 50, 70, 85, 101]
labels = ["<30", "30-50", "50-70", "70-85", ">=85"]
closed["st_bucket"] = pd.cut(closed["st_score"], bins=bins, labels=labels, right=False)
closed["sig_bucket"] = pd.cut(closed["signal_score"], bins=bins, labels=labels, right=False)

print("\n--- Win rate by st_score bucket ---")
gb = closed.groupby("st_bucket", observed=True)["hit"].agg(["sum", "count", "mean"])
gb.columns = ["targets", "total", "win_rate"]
gb["win_rate"] = (gb["win_rate"] * 100).round(1)
print(gb.to_string())

print("\n--- Win rate by signal_score bucket (baseline comparison) ---")
gb2 = closed.groupby("sig_bucket", observed=True)["hit"].agg(["sum", "count", "mean"])
gb2.columns = ["targets", "total", "win_rate"]
gb2["win_rate"] = (gb2["win_rate"] * 100).round(1)
print(gb2.to_string())

print("\n--- High-score subgroup detail ---")
for thresh, label in [(80, "st_score>=80"), (85, "st_score>=85"), (90, "st_score>=90")]:
    sub = closed[closed["st_score"] >= thresh]
    wr = sub["hit"].mean() * 100 if len(sub) else 0
    print(f"  {label}: n={len(sub):4d}  win_rate={wr:.1f}%")

for thresh, label in [(80, "signal_score>=80"), (85, "signal_score>=85"), (90, "signal_score>=90")]:
    sub = closed[closed["signal_score"] >= thresh]
    wr = sub["hit"].mean() * 100 if len(sub) else 0
    print(f"  {label}: n={len(sub):4d}  win_rate={wr:.1f}%")

# Score separation – mean st_score for winners vs losers
print("\n--- Score separation (winner vs loser st_score mean) ---")
winners = closed[closed["hit"] == 1]["st_score"]
losers  = closed[closed["hit"] == 0]["st_score"]
print(f"  st_score  winners mean: {winners.mean():.1f}  losers mean: {losers.mean():.1f}  delta: {winners.mean()-losers.mean():.1f}")
sig_win = closed[closed["hit"] == 1]["signal_score"]
sig_los = closed[closed["hit"] == 0]["signal_score"]
print(f"  sig_score winners mean: {sig_win.mean():.1f}  losers mean: {sig_los.mean():.1f}  delta: {sig_win.mean()-sig_los.mean():.1f}")
