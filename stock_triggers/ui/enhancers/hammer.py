"""Hammer enhancer.

Legacy hammer detection only checked candle shape. The default detector now
looks for the hammer shape plus context confirmation. In practice, requiring
all confirmation signals on the same bar was too strict, so the confirmed
version fires when the hammer shape is present and at least two of these are
true:

- RSI shows recent oversold context
- price is testing recent support
- volume is above its recent average

The older shape-only detector is still exposed via ``check_basic`` so its
historical impact can be compared against the confirmed version.
"""

from __future__ import annotations

import pandas as pd


def _select_history(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return prices.loc[prices["Ticker"] == ticker].sort_values("Date").copy()


def _is_hammer_bar(
    bar: pd.Series,
    *,
    body_pct_max: float,
    lower_shadow_min_pct: float,
    upper_shadow_max_pct: float,
) -> bool:
    o = float(bar["Open"])
    h = float(bar["High"])
    l = float(bar["Low"])
    c = float(bar["Close"])
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return (
        body / rng <= body_pct_max
        and lower_shadow / rng >= lower_shadow_min_pct
        and upper_shadow / rng <= upper_shadow_max_pct
    )


def _compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def check_basic(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.40,
    lower_shadow_min_pct: float = 0.50,
    upper_shadow_max_pct: float = 0.20,
) -> bool:
    """True if *ticker* shows a shape-only hammer in its last *lookback* bars."""
    g = _select_history(prices, ticker).tail(lookback)
    for _, bar in g.iterrows():
        if _is_hammer_bar(
            bar,
            body_pct_max=body_pct_max,
            lower_shadow_min_pct=lower_shadow_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
        ):
            return True
    return False


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.40,
    lower_shadow_min_pct: float = 0.50,
    upper_shadow_max_pct: float = 0.20,
    rsi_period: int = 14,
    rsi_oversold_max: float = 40.0,
    recent_oversold_lookback: int = 10,
    support_lookback: int = 20,
    support_tolerance_pct: float = 4.0,
    volume_avg_period: int = 20,
    volume_spike_ratio: float = 1.2,
    min_confirmation_count: int = 2,
    require_confirmations: bool = True,
) -> bool:
    """True if *ticker* shows a confirmed hammer in its last *lookback* bars."""
    history = _select_history(prices, ticker)
    if history.empty:
        return False

    recent = history.tail(lookback)
    if not require_confirmations:
        return check_basic(
            history,
            ticker,
            lookback=lookback,
            body_pct_max=body_pct_max,
            lower_shadow_min_pct=lower_shadow_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
        )

    working = history.copy()
    working["hammer_shape"] = working.apply(
        lambda bar: _is_hammer_bar(
            bar,
            body_pct_max=body_pct_max,
            lower_shadow_min_pct=lower_shadow_min_pct,
            upper_shadow_max_pct=upper_shadow_max_pct,
        ),
        axis=1,
    )
    working["rsi_value"] = _compute_rsi_series(working["Close"].astype(float), period=int(rsi_period))
    support_anchor = working["Low"].shift(1).rolling(int(support_lookback), min_periods=max(5, int(support_lookback) // 2)).min()
    volume_avg = working["Volume"].shift(1).rolling(int(volume_avg_period), min_periods=max(5, int(volume_avg_period) // 2)).mean()

    support_limit = support_anchor * (1.0 + float(support_tolerance_pct) / 100.0)
    working["near_support"] = support_anchor.notna() & (working["Low"].astype(float) <= support_limit)
    working["volume_spike"] = volume_avg.notna() & (
        working["Volume"].astype(float) >= (volume_avg * float(volume_spike_ratio))
    )
    working["recent_rsi_oversold"] = (
        working["rsi_value"]
        .rolling(int(recent_oversold_lookback), min_periods=1)
        .min()
        .le(float(rsi_oversold_max))
    )
    confirm_cols = ["recent_rsi_oversold", "near_support", "volume_spike"]
    working["confirmation_count"] = working[confirm_cols].sum(axis=1)
    working["confirmed_hammer"] = working["hammer_shape"] & (
        working["confirmation_count"] >= int(min_confirmation_count)
    )

    recent_dates = set(recent["Date"].tolist())
    return bool(working.loc[working["Date"].isin(recent_dates), "confirmed_hammer"].any())
