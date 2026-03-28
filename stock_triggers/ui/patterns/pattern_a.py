"""Pattern A – Trend Breakout with Volume.

Detects when price closes above a recent N-day high close on above-average
volume while the stock is in a confirmed uptrend (SMA50 > SMA200).
"""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    breakout_buffer_pct: float = 0.0,
    use_atr_stop: bool = False,
    atr_period: int = 14,
    atr_multiplier: float = 2.5,
) -> pd.DataFrame:
    """Return a DataFrame of Pattern A signals for *as_of_date*."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["PrevNHighClose"] = g["Close"].shift(1).rolling(breakout_days).max()
        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift(1)).abs()
        tr3 = (g["Low"] - g["Close"].shift(1)).abs()
        g["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g["ATR"] = g["TR"].rolling(int(atr_period)).mean()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["PrevNHighClose"]]
        if use_atr_stop:
            needed.append(r["ATR"])
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool(r["SMA50"] > r["SMA200"])
        cond_price = bool((r["Close"] > r["SMA50"]) and (r["Close"] > r["SMA200"]))
        breakout_level = float(r["PrevNHighClose"]) * (1.0 + float(breakout_buffer_pct) / 100.0)
        cond_breakout = bool(r["Close"] > breakout_level)
        cond_volume = bool(r["Volume"] >= volume_multiplier * r["VolAvg20"])

        if not (cond_trend and cond_price and cond_breakout and cond_volume):
            continue

        entry_price = float(r["Close"])
        if use_atr_stop:
            stop_price = entry_price - float(r["ATR"]) * float(atr_multiplier)
            stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0
        else:
            stop_price = entry_price * (1.0 - stop_pct / 100.0)
            stop_pct_eff = float(stop_pct)
        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": (
                    f"A_plus_breakout_{breakout_days}d"
                    if use_atr_stop or float(breakout_buffer_pct) > 0
                    else f"A_breakout_{breakout_days}d"
                ),
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
            }
        )

    cols = [
        "signal_date",
        "ticker",
        "pattern",
        "entry_price",
        "stop_pct",
        "stop_price",
    ]
    if not all_rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
    out = pd.DataFrame(all_rows, columns=cols)
    out["pattern_family"] = "A"
    out["score_trend"] = pd.NA
    out["score_setup"] = pd.NA
    out["score_volume"] = pd.NA
    out["score_rsi"] = pd.NA
    out["score_risk"] = pd.NA
    out["signal_score"] = pd.NA
    out["consensus_count"] = 1
    return out
