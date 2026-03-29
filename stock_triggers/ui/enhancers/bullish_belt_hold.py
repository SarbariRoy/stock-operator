"""Bullish Belt Hold enhancer.

Single-candle bullish reversal / continuation shape:
  opens near the low,
  closes strong near the high,
  long bullish real body.
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_min: float = 0.75,
    lower_shadow_max_pct: float = 0.05,
    upper_shadow_max_pct: float = 0.10,
    close_in_top_pct_max: float = 0.12,
    require_prev_bearish: bool = True,
) -> bool:
    """True if *ticker* shows a bullish belt hold in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 2))
    if g.empty:
        return False

    bar = g.iloc[-1]
    o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    rng = h - l
    if rng <= 0 or c <= o:
        return False

    if require_prev_bearish and len(g) >= 2:
        prev = g.iloc[-2]
        if float(prev["Close"]) >= float(prev["Open"]):
            return False

    body = c - o
    lower_shadow = o - l
    upper_shadow = h - c
    close_from_high = h - c
    return (
        body / rng >= body_pct_min
        and lower_shadow / rng <= lower_shadow_max_pct
        and upper_shadow / rng <= upper_shadow_max_pct
        and close_from_high / rng <= close_in_top_pct_max
    )
