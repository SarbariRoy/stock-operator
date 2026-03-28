"""Dragonfly Doji enhancer.

T-shaped candle: tiny body at the top, long lower shadow, minimal upper shadow.
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    body_pct_max: float = 0.30,
    upper_shadow_max_pct: float = 0.15,
) -> bool:
    """True if *ticker* shows a dragonfly-doji candle in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(lookback)
    for _, bar in g.iterrows():
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        rng = h - l
        if rng <= 0:
            continue
        if (
            abs(c - o) / rng <= body_pct_max
            and (min(o, c) - l) / rng >= 0.60
            and (h - max(o, c)) / rng <= upper_shadow_max_pct
        ):
            return True
    return False
