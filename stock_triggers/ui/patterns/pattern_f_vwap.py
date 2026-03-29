"""Pattern F – VWAP Reclaim.

Price crosses above an approximated VWAP (volume-weighted average price over
recent N bars) with a volume spike while SMA50 > SMA200.

Because we only have EOD data, VWAP is approximated as the rolling
volume-weighted typical price:  TP = (High + Low + Close) / 3.
"""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS
from .scoring import apply_ma_slope_bonus, build_score_components, compute_ma_slope_pct


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float = 1.2,
    stop_pct: float = 7.0,
    vwap_period: int = 20,
    compute_rsi_fn=None,
) -> pd.DataFrame:
    """Return a DataFrame of VWAP reclaim signals for *as_of_date*."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["TP"] = (g["High"] + g["Low"] + g["Close"]) / 3.0
        g["TP_Vol"] = g["TP"] * g["Volume"]
        g["VWAP"] = (
            g["TP_Vol"].rolling(vwap_period).sum()
            / g["Volume"].rolling(vwap_period).sum()
        )
        g["ClosePrev1"] = g["Close"].shift(1)
        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["SwingLow10"] = g["Low"].shift(1).rolling(10).min()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["VWAP"], r["ClosePrev1"], r["SwingLow10"]]
        if any(pd.isna(v) for v in needed):
            continue

        # Trend filter
        if not float(r["SMA50"]) > float(r["SMA200"]):
            continue

        # VWAP reclaim: previous close was below VWAP, today close is above
        if not (float(r["ClosePrev1"]) <= float(r["VWAP"]) and float(r["Close"]) > float(r["VWAP"])):
            continue

        # Volume spike
        if float(r["VolAvg20"]) > 0:
            vol_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        else:
            continue
        if vol_ratio < float(volume_multiplier):
            continue

        entry_price = float(r["Close"])
        vwap_stop = float(r["VWAP"]) * 0.99  # just below VWAP
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        stop_price = max(fixed_stop, vwap_stop, float(r["SwingLow10"]))
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        # Setup strength: how far price reclaimed above VWAP
        reclaim_pct = ((float(r["Close"]) / float(r["VWAP"])) - 1.0) * 100.0
        setup_strength_pct = reclaim_pct * 5.0  # scale for scoring

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
                "pattern": "F_vwap_reclaim",
                "pattern_family": "F",
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
