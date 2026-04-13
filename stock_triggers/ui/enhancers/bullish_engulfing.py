"""Bullish Engulfing enhancer.

The live detector remains the legacy shape-only definition because the stricter
confirmed variant has not yet earned a positive historical edge. This module
therefore exposes both:

- ``check_basic`` / ``check``: textbook two-candle engulfing geometry
- ``check_confirmed``: experimental context-aware engulfing for calibration
"""

from __future__ import annotations

import math

import pandas as pd


def _select_history(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return prices.loc[prices["Ticker"] == ticker].sort_values("Date").copy()


def _is_bullish_engulfing_pair(prev: pd.Series, curr: pd.Series) -> bool:
    o_prev = float(prev["Open"])
    c_prev = float(prev["Close"])
    o_curr = float(curr["Open"])
    c_curr = float(curr["Close"])

    if not c_prev < o_prev:
        return False
    if not c_curr > o_curr:
        return False
    return o_curr <= c_prev and c_curr >= o_prev


def _body_size(bar: pd.Series) -> float:
    return abs(float(bar["Close"]) - float(bar["Open"]))


def check_basic(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
) -> bool:
    """True if *ticker* shows a shape-only bullish engulfing in recent bars."""
    g = _select_history(prices, ticker).tail(max(lookback, 2))
    if len(g) < 2:
        return False

    rows = list(g.iterrows())
    for i in range(len(rows) - 1):
        _, prev = rows[i]
        _, curr = rows[i + 1]
        if _is_bullish_engulfing_pair(prev, curr):
            return True
    return False


def check_confirmed(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    trend_fast_period: int = 50,
    trend_slow_period: int = 200,
    support_lookback: int = 20,
    support_tolerance_pct: float = 3.0,
    pullback_lookback: int = 15,
    pullback_min_pct: float = 2.0,
    consolidation_lookback: int = 10,
    consolidation_range_max_pct: float = 8.0,
    volume_avg_period: int = 20,
    volume_spike_ratio: float = 1.25,
    max_gap_down_pct: float = 2.0,
    body_size_ratio_min: float = 1.1,
    close_through_prev_high: bool = True,
    upper_shadow_max_pct: float = 0.25,
        min_confirmation_count: int = 4,
    require_confirmations: bool = True,
) -> bool:
    """True if *ticker* shows a confirmed bullish engulfing in recent bars."""
    history = _select_history(prices, ticker)
    if len(history) < 2:
        return False

    if not require_confirmations:
        return check_basic(history, ticker, lookback=lookback)

    working = history.copy()
    working["sma_fast"] = working["Close"].astype(float).rolling(int(trend_fast_period)).mean()
    working["sma_slow"] = working["Close"].astype(float).rolling(int(trend_slow_period)).mean()

    prior_low = working["Low"].shift(1).rolling(
        int(support_lookback), min_periods=max(5, int(support_lookback) // 2)
    ).min()
    prior_swing_high = working["Close"].shift(1).rolling(
        int(pullback_lookback), min_periods=max(5, int(pullback_lookback) // 2)
    ).max()
    prior_range_high = working["High"].shift(1).rolling(
        int(consolidation_lookback), min_periods=max(5, int(consolidation_lookback) // 2)
    ).max()
    prior_range_low = working["Low"].shift(1).rolling(
        int(consolidation_lookback), min_periods=max(5, int(consolidation_lookback) // 2)
    ).min()
    prior_close = working["Close"].shift(1)
    volume_avg = working["Volume"].shift(1).rolling(
        int(volume_avg_period), min_periods=max(5, int(volume_avg_period) // 2)
    ).mean()

    pair_rows: list[dict[str, object]] = []
    rows = list(working.iterrows())
    for i in range(len(rows) - 1):
        prev_idx, prev = rows[i]
        curr_idx, curr = rows[i + 1]

        pair_rows.append(
            {
                "index": curr_idx,
                "engulfing_shape": _is_bullish_engulfing_pair(prev, curr),
                "prev_close": float(prev["Close"]),
                "prev_high": float(prev["High"]),
                "prev_body": _body_size(prev),
            }
        )

    pair_df = pd.DataFrame(pair_rows).set_index("index") if pair_rows else pd.DataFrame()
    working["engulfing_shape"] = False
    working["prev_close"] = math.nan
    working["prev_high"] = math.nan
    working["prev_body"] = math.nan
    if not pair_df.empty:
        working.loc[pair_df.index, "engulfing_shape"] = pair_df["engulfing_shape"].astype(bool)
        working.loc[pair_df.index, "prev_close"] = pair_df["prev_close"].astype(float)
        working.loc[pair_df.index, "prev_high"] = pair_df["prev_high"].astype(float)
        working.loc[pair_df.index, "prev_body"] = pair_df["prev_body"].astype(float)

    support_limit = prior_low * (1.0 + float(support_tolerance_pct) / 100.0)
    near_support = prior_low.notna() & (working["Low"].astype(float) <= support_limit)
    pulled_back = prior_swing_high.notna() & (
        prior_swing_high >= (working["Close"].astype(float) * (1.0 + float(pullback_min_pct) / 100.0))
    )

    prior_range_pct = ((prior_range_high - prior_range_low) / prior_close.replace(0.0, float("nan"))) * 100.0
    working["trend_context"] = (
        working["sma_fast"].notna()
        & working["sma_slow"].notna()
        & (working["sma_fast"] > working["sma_slow"])
        & (working["Close"].astype(float) >= working["sma_fast"])
    )
    working["support_or_pullback_context"] = near_support | (pulled_back & working["trend_context"])
    working["recent_consolidation"] = prior_range_pct.le(float(consolidation_range_max_pct))
    working["volume_spike"] = volume_avg.notna() & (
        working["Volume"].astype(float) >= (volume_avg * float(volume_spike_ratio))
    )
    working["non_extreme_gap"] = working["prev_close"].notna() & (
        working["Open"].astype(float) >= (working["prev_close"].astype(float) * (1.0 - float(max_gap_down_pct) / 100.0))
    )
    current_body = (working["Close"].astype(float) - working["Open"].astype(float)).abs()
    current_range = working["High"].astype(float) - working["Low"].astype(float)
    upper_shadow = working["High"].astype(float) - working[["Open", "Close"]].astype(float).max(axis=1)
    working["body_dominance"] = working["prev_body"].notna() & working["prev_body"].gt(0.0) & (
        current_body >= (working["prev_body"].astype(float) * float(body_size_ratio_min))
    )
    working["close_strength"] = current_range.gt(0.0) & (
        upper_shadow / current_range <= float(upper_shadow_max_pct)
    )
    if close_through_prev_high:
        working["close_strength"] = working["close_strength"] & (
            working["prev_high"].notna() & (working["Close"].astype(float) >= working["prev_high"].astype(float))
        )

    confirm_cols = [
        "trend_context",
        "support_or_pullback_context",
        "recent_consolidation",
        "volume_spike",
        "non_extreme_gap",
    ]
    working["confirmation_count"] = working[confirm_cols].sum(axis=1)
    working["confirmed_engulfing"] = working["engulfing_shape"] & working["body_dominance"] & working["close_strength"] & (
        working["confirmation_count"] >= int(min_confirmation_count)
    )

    recent = working.tail(max(lookback, 2))
    recent_dates = set(recent["Date"].iloc[1:].tolist())
    return bool(working.loc[working["Date"].isin(recent_dates), "confirmed_engulfing"].any())


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
) -> bool:
    """True if *ticker* shows the live shape-only bullish engulfing pattern."""
    return check_basic(prices, ticker, lookback=lookback)
