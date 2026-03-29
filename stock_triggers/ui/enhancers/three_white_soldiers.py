"""Three White Soldiers enhancer.

Three-candle bullish reversal / continuation pattern:
  three consecutive bullish candles,
  each with meaningful body,
  each closing above the prior close.
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 3,
    body_pct_min: float = 0.50,
    upper_shadow_max_pct: float = 0.20,
    min_total_gain_pct: float = 0.04,
    require_higher_highs: bool = True,
    require_higher_closes: bool = True,
) -> bool:
    """True if *ticker* shows three white soldiers in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 3))
    if len(g) < 3:
        return False

    bars = list(g.tail(3).itertuples(index=False))
    prev_close = None
    prev_high = None
    first_open = float(bars[0].Open)
    for bar in bars:
        o, h, l, c = float(bar.Open), float(bar.High), float(bar.Low), float(bar.Close)
        rng = h - l
        if rng <= 0:
            return False
        body = c - o
        upper_shadow = h - c
        if not (c > o and body / rng >= body_pct_min and upper_shadow / rng <= upper_shadow_max_pct):
            return False
        if prev_close is not None and require_higher_closes and c <= prev_close:
            return False
        if prev_high is not None and require_higher_highs and h <= prev_high:
            return False
        prev_close = c
        prev_high = h
    return (float(bars[-1].Close) / first_open - 1.0) >= min_total_gain_pct
