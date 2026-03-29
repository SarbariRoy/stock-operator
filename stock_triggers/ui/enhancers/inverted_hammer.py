"""Inverted Hammer enhancer.

Single-candle bullish reversal shape:
  small real body near the low of the range,
  long upper shadow,
  minimal lower shadow.
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.40,
    upper_shadow_min_pct: float = 0.45,
    lower_shadow_max_pct: float = 0.20,
) -> bool:
    """True if *ticker* shows an inverted hammer in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(lookback)
    for _, bar in g.iterrows():
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        rng = h - l
        if rng <= 0:
            continue
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        if (
            body / rng <= body_pct_max
            and upper_shadow / rng >= upper_shadow_min_pct
            and lower_shadow / rng <= lower_shadow_max_pct
        ):
            return True
    return False
