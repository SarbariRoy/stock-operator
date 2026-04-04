"""Bullish Marubozu enhancer.

This implementation is practical rather than textbook-only. It looks for a
strong green marubozu-style candle and then asks for contextual confirmation
from recent consolidation, a strong location signal, and volume expansion.

Breakout-level and pullback-support are treated as alternatives because they
describe different trade locations.
"""

from __future__ import annotations

import pandas as pd


def _select_history(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return prices.loc[prices["Ticker"] == ticker].sort_values("Date").copy()


def _is_bullish_marubozu_bar(
    bar: pd.Series,
    *,
    body_min_pct: float,
    upper_shadow_max_pct: float,
    lower_shadow_max_pct: float,
) -> bool:
    o = float(bar["Open"])
    h = float(bar["High"])
    l = float(bar["Low"])
    c = float(bar["Close"])
    rng = h - l
    if rng <= 0 or c <= o:
        return False
    body = c - o
    upper_shadow = h - c
    lower_shadow = o - l
    return (
        body / rng >= body_min_pct
        and upper_shadow / rng <= upper_shadow_max_pct
        and lower_shadow / rng <= lower_shadow_max_pct
    )


def check_basic(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 1,
    body_min_pct: float = 0.80,
    upper_shadow_max_pct: float = 0.10,
    lower_shadow_max_pct: float = 0.10,
) -> bool:
    """True if *ticker* shows a shape-only bullish marubozu in recent bars."""
    recent = _select_history(prices, ticker).tail(max(lookback, 1))
    for _, bar in recent.iterrows():
        if _is_bullish_marubozu_bar(
            bar,
            body_min_pct=body_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
            lower_shadow_max_pct=lower_shadow_max_pct,
        ):
            return True
    return False


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 1,
    body_min_pct: float = 0.80,
    upper_shadow_max_pct: float = 0.10,
    lower_shadow_max_pct: float = 0.10,
    consolidation_lookback: int = 12,
    consolidation_range_max_pct: float = 7.0,
    breakout_lookback: int = 20,
    breakout_tolerance_pct: float = 0.5,
    support_lookback: int = 20,
    support_tolerance_pct: float = 3.0,
    pullback_min_pct: float = 2.0,
    volume_avg_period: int = 20,
    volume_spike_ratio: float = 1.5,
    min_confirmation_count: int = 2,
    require_confirmations: bool = True,
) -> bool:
    """True if *ticker* shows a confirmed bullish marubozu in recent bars."""
    history = _select_history(prices, ticker)
    if history.empty:
        return False

    recent = history.tail(max(lookback, 1))
    if not require_confirmations:
        return check_basic(
            history,
            ticker,
            lookback=lookback,
            body_min_pct=body_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
            lower_shadow_max_pct=lower_shadow_max_pct,
        )

    working = history.copy()
    working["marubozu_shape"] = working.apply(
        lambda bar: _is_bullish_marubozu_bar(
            bar,
            body_min_pct=body_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
            lower_shadow_max_pct=lower_shadow_max_pct,
        ),
        axis=1,
    )

    prior_close = working["Close"].shift(1)
    prior_high = working["High"].shift(1).rolling(int(breakout_lookback), min_periods=max(5, int(breakout_lookback) // 2)).max()
    prior_low = working["Low"].shift(1).rolling(int(support_lookback), min_periods=max(5, int(support_lookback) // 2)).min()
    prior_range_high = working["High"].shift(1).rolling(int(consolidation_lookback), min_periods=max(5, int(consolidation_lookback) // 2)).max()
    prior_range_low = working["Low"].shift(1).rolling(int(consolidation_lookback), min_periods=max(5, int(consolidation_lookback) // 2)).min()
    volume_avg = working["Volume"].shift(1).rolling(int(volume_avg_period), min_periods=max(5, int(volume_avg_period) // 2)).mean()
    prior_swing_high = working["Close"].shift(1).rolling(int(support_lookback), min_periods=max(5, int(support_lookback) // 2)).max()

    prior_range_pct = ((prior_range_high - prior_range_low) / prior_close.replace(0.0, float("nan"))) * 100.0
    working["recent_consolidation"] = prior_range_pct.le(float(consolidation_range_max_pct))

    breakout_trigger = prior_high * (1.0 - float(breakout_tolerance_pct) / 100.0)
    working["near_breakout_level"] = prior_high.notna() & (working["Close"].astype(float) >= breakout_trigger)

    support_limit = prior_low * (1.0 + float(support_tolerance_pct) / 100.0)
    near_support = prior_low.notna() & (working["Low"].astype(float) <= support_limit)
    pulled_back = prior_swing_high.notna() & (
        prior_swing_high >= (working["Close"].astype(float) * (1.0 + float(pullback_min_pct) / 100.0))
    )
    working["pullback_support_context"] = near_support & pulled_back
    working["context_location"] = working["near_breakout_level"] | working["pullback_support_context"]

    working["volume_spike"] = volume_avg.notna() & (
        working["Volume"].astype(float) >= (volume_avg * float(volume_spike_ratio))
    )

    confirm_cols = ["recent_consolidation", "context_location", "volume_spike"]
    working["confirmation_count"] = working[confirm_cols].sum(axis=1)
    working["confirmed_marubozu"] = working["marubozu_shape"] & (
        working["confirmation_count"] >= int(min_confirmation_count)
    )

    recent_dates = set(recent["Date"].tolist())
    return bool(working.loc[working["Date"].isin(recent_dates), "confirmed_marubozu"].any())