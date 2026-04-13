"""Analyze bullish engulfing performance by pattern family.

Reports live shape-only engulfing and the experimental confirmed engulfing
variant against each pattern-family baseline so family-specific rules can be
evaluated before promoting them into the live detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stock_triggers.ui.enhancers import bullish_engulfing

ENGULFING_POSITIVE_FAMILIES = {"A", "C", "G"}


def main() -> None:
    prices = pd.read_csv("stock_triggers/data/prices_eod.csv", parse_dates=["Date"])
    signals = pd.read_csv("stock_triggers/data/signals_all_patterns.csv")
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])

    grouped = {str(ticker): group.sort_values("Date") for ticker, group in prices.groupby("Ticker", sort=False)}
    rows: list[dict[str, object]] = []

    for _, sig in signals.iterrows():
        ticker = str(sig["ticker"])
        signal_date = pd.to_datetime(sig["signal_date"])
        entry = pd.to_numeric(sig.get("entry_price"), errors="coerce")
        if pd.isna(entry) or float(entry) <= 0:
            continue

        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        history = grouped.get(ticker_ns)
        if history is None:
            continue

        upto_signal = history[history["Date"] <= signal_date]
        if upto_signal.empty:
            continue

        future = history[history["Date"] > signal_date].head(30)
        take_profit = float(entry) * 1.06
        stop_price = float(entry) * 0.93
        outcome = "hold"
        for _, bar in future.iterrows():
            if float(bar["High"]) >= take_profit:
                outcome = "win"
                break
            if float(bar["Low"]) <= stop_price:
                outcome = "loss"
                break

        rows.append(
            {
                "pattern_family": str(sig.get("pattern_family", "")).strip().upper(),
                "outcome": outcome,
                "engulfing_live": bullish_engulfing.check(upto_signal, ticker_ns),
                "engulfing_trial": bullish_engulfing.check_confirmed(upto_signal, ticker_ns),
            }
        )

    df = pd.DataFrame(rows)
    df["engulfing_trend_combo"] = (
        df["engulfing_live"].astype(bool)
        & df["pattern_family"].isin(ENGULFING_POSITIVE_FAMILIES)
    )
    overall_wr = (df["outcome"] == "win").mean()
    print(f"overall baseline win rate: {overall_wr * 100:.1f}% on {len(df)} rows")
    print()

    for family in ["A", "B", "C", "D", "E", "F", "G"]:
        family_df = df[df["pattern_family"] == family]
        family_wr = (family_df["outcome"] == "win").mean() if not family_df.empty else 0.0
        print(f"Family {family}: n={len(family_df)} base_wr={family_wr * 100:.1f}%")
        for column in ("engulfing_live", "engulfing_trial"):
            subset = family_df[family_df[column] == True]  # noqa: E712
            if subset.empty:
                print(f"  {column}: count=0")
                continue
            subset_wr = (subset["outcome"] == "win").mean()
            edge = (subset_wr - family_wr) * 100.0
            print(
                f"  {column}: count={len(subset)} wr={subset_wr * 100:.1f}% edge_vs_family={edge:.1f}pp"
            )
        print()

    combo_df = df[df["pattern_family"].isin(ENGULFING_POSITIVE_FAMILIES)].copy()
    combo_base_wr = (combo_df["outcome"] == "win").mean()
    combo_hits = combo_df[combo_df["engulfing_trend_combo"] == True]  # noqa: E712
    print(
        f"Combo A/C/G baseline: n={len(combo_df)} base_wr={combo_base_wr * 100:.1f}% | "
        f"engulfing_trend_combo count={len(combo_hits)} wr={((combo_hits['outcome'] == 'win').mean() * 100.0) if not combo_hits.empty else 0.0:.1f}%"
    )


if __name__ == "__main__":
    main()