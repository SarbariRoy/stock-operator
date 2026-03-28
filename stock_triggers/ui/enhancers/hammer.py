"""Hammer enhancer.

Small real body at the upper end of the candle, long lower shadow (>= 50%
of range), minimal upper shadow.  Similar to dragonfly doji but allows a
larger body (up to 40% of range).
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.40,
    lower_shadow_min_pct: float = 0.50,
    upper_shadow_max_pct: float = 0.20,
) -> bool:
    """True if *ticker* shows a hammer candle in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(lookback)
    for _, bar in g.iterrows():
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        rng = h - l
        if rng <= 0:
            continue
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        if (
            body / rng <= body_pct_max
            and lower_shadow / rng >= lower_shadow_min_pct
            and upper_shadow / rng <= upper_shadow_max_pct
        ):
            return True
    return False
