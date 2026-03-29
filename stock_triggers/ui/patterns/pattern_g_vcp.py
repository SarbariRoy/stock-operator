"""Pattern G – Volatility Contraction Pattern (VCP).

Detector based on explicit pullback-depth contraction:
- established uptrend
- at least three pivot-high to pivot-low pullbacks
- each recent pullback shallower than the last
- breakout above recent resistance on volume
"""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS
from .scoring import apply_ma_slope_bonus, build_score_components, compute_ma_slope_pct


def _find_price_pivots(hist: pd.DataFrame, *, span: int) -> list[dict]:
    pivots: list[dict] = []
    if len(hist) < (span * 2) + 1:
        return pivots

    highs = hist["High"].astype(float).reset_index(drop=True)
    lows = hist["Low"].astype(float).reset_index(drop=True)
    dates = hist["Date"].reset_index(drop=True)

    for i in range(span, len(hist) - span):
        hi = float(highs.iloc[i])
        lo = float(lows.iloc[i])
        prev_hi = highs.iloc[i - span:i]
        next_hi = highs.iloc[i + 1:i + span + 1]
        prev_lo = lows.iloc[i - span:i]
        next_lo = lows.iloc[i + 1:i + span + 1]

        if hi > float(prev_hi.max()) and hi >= float(next_hi.max()):
            pivots.append({"kind": "high", "idx": i, "date": dates.iloc[i], "price": hi})
        if lo < float(prev_lo.min()) and lo <= float(next_lo.min()):
            pivots.append({"kind": "low", "idx": i, "date": dates.iloc[i], "price": lo})

    pivots.sort(key=lambda item: item["idx"])
    return pivots


def _extract_pullbacks_from_pivots(pivots: list[dict]) -> list[dict]:
    pullbacks: list[dict] = []
    i = 0
    while i < len(pivots):
        pivot = pivots[i]
        if pivot["kind"] != "high":
            i += 1
            continue

        next_high_idx = None
        for j in range(i + 1, len(pivots)):
            if pivots[j]["kind"] == "high":
                next_high_idx = j
                break

        candidate_lows = [
            p for p in pivots[i + 1:(next_high_idx if next_high_idx is not None else len(pivots))]
            if p["kind"] == "low" and p["idx"] > pivot["idx"]
        ]
        if candidate_lows:
            low_pivot = min(candidate_lows, key=lambda item: item["price"])
            depth_pct = ((float(pivot["price"]) - float(low_pivot["price"])) / float(pivot["price"])) * 100.0
            pullbacks.append(
                {
                    "high_idx": pivot["idx"],
                    "high_date": pivot["date"],
                    "high_price": float(pivot["price"]),
                    "low_idx": low_pivot["idx"],
                    "low_date": low_pivot["date"],
                    "low_price": float(low_pivot["price"]),
                    "depth_pct": float(depth_pct),
                }
            )
        i += 1
    return pullbacks


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float = 1.0,
    stop_pct: float = 7.0,
    pivot_span: int = 3,
    base_lookback: int = 80,
    breakout_lookback: int = 20,
    dryup_volume_ratio: float = 0.9,
    compute_rsi_fn=None,
) -> pd.DataFrame:
    """Return a DataFrame of VCP-style breakout signals for *as_of_date*."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["High20"] = g["High"].shift(1).rolling(breakout_lookback).max()
        g["Low10"] = g["Low"].shift(1).rolling(10).min()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["High20"], r["Low10"]]
        if any(pd.isna(v) for v in needed):
            continue

        hist = g[g["Date"] <= as_of_date].tail(base_lookback + 1).copy()
        if len(hist) < max(50, (pivot_span * 2) + 7):
            continue
        pre_breakout = hist.iloc[:-1].copy()
        pivots = _find_price_pivots(pre_breakout[["Date", "High", "Low"]].copy(), span=int(pivot_span))
        pullbacks = _extract_pullbacks_from_pivots(pivots)
        if len(pullbacks) < 3:
            continue

        recent_pullbacks = pullbacks[-3:]
        depth1 = float(recent_pullbacks[0]["depth_pct"])
        depth2 = float(recent_pullbacks[1]["depth_pct"])
        depth3 = float(recent_pullbacks[2]["depth_pct"])
        contraction_ok = (
            depth1 > depth2 > depth3
            and depth2 <= depth1 * 0.9
            and depth3 <= depth2 * 0.9
        )
        if not contraction_ok:
            continue

        last_pullback = recent_pullbacks[-1]
        if int(last_pullback["low_idx"]) >= len(pre_breakout) - 1:
            continue

        cond_trend = bool((float(r["SMA50"]) > float(r["SMA200"])) and (float(r["Close"]) > float(r["SMA50"])))
        recent_resistance = max(float(r["High20"]), max(float(p["high_price"]) for p in recent_pullbacks))
        cond_breakout = bool(float(r["Close"]) > recent_resistance)
        vol_ratio = float(r["Volume"]) / float(r["VolAvg20"]) if float(r["VolAvg20"]) > 0 else 0.0
        cond_volume = bool(vol_ratio >= float(volume_multiplier))
        dryup_slice = pre_breakout.iloc[int(last_pullback["high_idx"]):int(last_pullback["low_idx"]) + 1].copy()
        recent_dryup_ratio = float(dryup_slice["Volume"].mean()) / float(r["VolAvg20"]) if (not dryup_slice.empty and float(r["VolAvg20"]) > 0) else 1.0
        cond_dryup = bool(recent_dryup_ratio <= float(dryup_volume_ratio))

        if not (cond_trend and cond_breakout and cond_volume and cond_dryup):
            continue

        entry_price = float(r["Close"])
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        contraction_low = float(last_pullback["low_price"])
        stop_price = min(entry_price * 0.995, max(fixed_stop, contraction_low, float(r["Low10"])))
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0 if entry_price > 0 else float(stop_pct)

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        setup_strength_pct = max(0.0, ((depth1 - depth3) / max(depth1, 0.01)) * 100.0)

        rsi_value = None
        if compute_rsi_fn is not None:
            try:
                hist_close = g[g["Date"] <= as_of_date]["Close"].astype(float)
                rsi_value = compute_rsi_fn(hist_close, period=14)
            except Exception:
                rsi_value = None

        score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=vol_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )
        sma50_slope_pct = compute_ma_slope_pct(g[g["Date"] <= as_of_date]["SMA50"])
        ma_slope_bonus, signal_score = apply_ma_slope_bonus(signal_score, sma50_slope_pct)

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": "G_vcp_breakout",
                "pattern_family": "G",
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(float(stop_price), 4),
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