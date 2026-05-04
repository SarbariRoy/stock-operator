"""Pattern B – Pullback & Rebound near SMA20 in an uptrend."""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS
from .scoring import apply_ma_slope_bonus, build_score_components, compute_ma_slope_pct


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float = 1.5,
    rebound_min_pct: float = 0.2,
    compute_rsi_fn=None,
    precomputed_features: bool = False,
    ticker_groups: "dict | None" = None,
) -> pd.DataFrame:
    """Return a DataFrame of Pattern B signals for *as_of_date*."""

    all_rows: list[dict] = []
    _iter = ticker_groups.items() if ticker_groups is not None else prices.groupby("Ticker", sort=True)
    for ticker, g in _iter:
        if not precomputed_features:
            g = g.copy().sort_values("Date")
            g["SMA20"] = g["Close"].rolling(20).mean()
            g["SMA50"] = g["Close"].rolling(50).mean()
            g["SMA200"] = g["Close"].rolling(200).mean()
            g["VolAvg20"] = g["Volume"].rolling(20).mean()
            g["ClosePrev1"] = g["Close"].shift(1)
            g["SwingLow10"] = g["Low"].shift(1).rolling(10).min()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [
            r["SMA20"],
            r["SMA50"],
            r["SMA200"],
            r["VolAvg20"],
            r["ClosePrev1"],
            r["SwingLow10"],
        ]
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool((r["SMA50"] > r["SMA200"]) and (r["Close"] > r["SMA50"]))
        cond_pullback = bool(r["Close"] <= float(r["SMA20"]) * (1.0 + float(pullback_buffer_pct) / 100.0))
        cond_rebound = bool(
            r["Close"] >= float(r["ClosePrev1"]) * (1.0 + float(rebound_min_pct) / 100.0)
        )
        cond_volume = bool(r["Volume"] >= max(1.0, float(volume_multiplier) * 0.8) * float(r["VolAvg20"]))

        if not (cond_trend and cond_pullback and cond_rebound and cond_volume):
            continue

        entry_price = float(r["Close"])
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        stop_price = min(entry_price * 0.995, max(fixed_stop, float(r["SwingLow10"])))
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        setup_strength_pct = ((float(r["SMA20"]) / float(r["Close"])) - 1.0) * 100.0
        volume_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        if precomputed_features and "RSI" in r.index and not pd.isna(r.get("RSI")):
            rsi_value = float(r["RSI"])
        elif compute_rsi_fn is not None:
            try:
                hist_close = g[g["Date"] <= as_of_date]["Close"].astype(float)
                rsi_value = compute_rsi_fn(hist_close, period=14)
            except Exception:
                rsi_value = None
        else:
            rsi_value = None

        score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=volume_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )
        if precomputed_features and "SMA50Slope5d" in r.index and not pd.isna(r.get("SMA50Slope5d")):
            sma50_slope_pct = float(r["SMA50Slope5d"]) if float(r["SMA50Slope5d"]) > 0 else None
        else:
            sma50_slope_pct = compute_ma_slope_pct(g[g["Date"] <= as_of_date]["SMA50"])
        ma_slope_bonus, signal_score = apply_ma_slope_bonus(signal_score, sma50_slope_pct)

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": "B_pullback_rebound",
                "pattern_family": "B",
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
                "score_trend": score_trend,
                "score_setup": score_setup,
                "score_volume": score_volume,
                "score_rsi": score_rsi,
                "score_risk": score_risk,
                "sma50_slope_pct": round(float(sma50_slope_pct), 2) if sma50_slope_pct is not None else pd.NA,
                "ma_slope_bonus": ma_slope_bonus,
                "signal_score": signal_score,
                "consensus_count": 1,
            }
        )

    if not all_rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
    return pd.DataFrame(all_rows, columns=STANDARD_SIGNAL_COLS)
