"""Pattern E – Bollinger Squeeze Breakout.

Bollinger Band(20,2) width reaches a 120-day low (squeeze), then price
breaks above the upper band on above-average volume in an uptrend.
"""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS
from .scoring import apply_ma_slope_bonus, build_score_components, compute_ma_slope_pct


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float = 1.0,
    stop_pct: float = 7.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    squeeze_lookback: int = 120,
    compute_rsi_fn=None,
) -> pd.DataFrame:
    """Return a DataFrame of Bollinger squeeze-breakout signals for *as_of_date*."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA_BB"] = g["Close"].rolling(bb_period).mean()
        g["BB_std"] = g["Close"].rolling(bb_period).std()
        g["BB_upper"] = g["SMA_BB"] + bb_std * g["BB_std"]
        g["BB_lower"] = g["SMA_BB"] - bb_std * g["BB_std"]
        g["BB_width"] = (g["BB_upper"] - g["BB_lower"]) / g["SMA_BB"]
        g["BB_width_min"] = g["BB_width"].rolling(squeeze_lookback).min()
        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [
            r["SMA50"], r["SMA200"], r["VolAvg20"],
            r["BB_upper"], r["BB_width"], r["BB_width_min"], r["SMA_BB"],
        ]
        if any(pd.isna(v) for v in needed):
            continue

        # Trend filter
        if not float(r["SMA50"]) > float(r["SMA200"]):
            continue

        # Squeeze condition: current BB width is within 5% of the recent minimum
        if float(r["BB_width"]) > float(r["BB_width_min"]) * 1.05:
            # Check if squeeze happened in last 3 bars (recent squeeze with breakout today)
            recent = g[g["Date"] <= as_of_date].tail(3)
            squeeze_recent = any(
                pd.notna(row2["BB_width"]) and pd.notna(row2["BB_width_min"])
                and float(row2["BB_width"]) <= float(row2["BB_width_min"]) * 1.05
                for _, row2 in recent.iterrows()
            )
            if not squeeze_recent:
                continue

        # Breakout: close above upper band
        if not float(r["Close"]) > float(r["BB_upper"]):
            continue

        # Volume filter
        if float(r["VolAvg20"]) > 0:
            vol_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        else:
            continue
        if vol_ratio < float(volume_multiplier) * 0.8:
            continue

        entry_price = float(r["Close"])
        # Stop at middle band (SMA_BB) or fixed, whichever is higher
        sma_stop = float(r["SMA_BB"])
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        stop_price = max(fixed_stop, sma_stop)
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        # Setup strength: squeeze tightness (narrower = stronger setup)
        squeeze_ratio = float(r["BB_width_min"]) / max(float(r["BB_width"]), 0.001)
        setup_strength_pct = squeeze_ratio * 8.0  # scale for scoring

        rsi_value = None
        if compute_rsi_fn is not None:
            try:
                hist_close = g[g["Date"] <= as_of_date]["Close"].astype(float)
                rsi_value = compute_rsi_fn(hist_close, period=14)
            except Exception:
                rsi_value = None

        scores = build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=vol_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )
        sma50_slope_pct = compute_ma_slope_pct(g[g["Date"] <= as_of_date]["SMA50"])
        ma_slope_bonus, boosted_signal_score = apply_ma_slope_bonus(scores[5], sma50_slope_pct)

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": "E_boll_squeeze",
                "pattern_family": "E",
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
                "score_trend": scores[0],
                "score_setup": scores[1],
                "score_volume": scores[2],
                "score_rsi": scores[4],
                "score_risk": scores[3],
                "sma50_slope_pct": round(float(sma50_slope_pct), 2) if sma50_slope_pct is not None else pd.NA,
                "ma_slope_bonus": ma_slope_bonus,
                "signal_score": boosted_signal_score,
                "consensus_count": 1,
            }
        )

    if not all_rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
    return pd.DataFrame(all_rows, columns=STANDARD_SIGNAL_COLS)
