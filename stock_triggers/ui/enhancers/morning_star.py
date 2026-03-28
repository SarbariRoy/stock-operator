"""Morning Star enhancer.

Three-candle bullish reversal pattern:
  bar[-3]: long red body (body > 50% of range, close < open)
  bar[-2]: small body (body < 30% of range) — the star
  bar[-1]: long green body (body > 40% of range, close > open) that closes
           above the midpoint of bar[-3]
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 3,
) -> bool:
    """True if *ticker* shows a morning-star pattern ending in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 3))
    if len(g) < 3:
        return False

    rows = list(g.itertuples(index=False))
    # Check consecutive triplets (usually just the last 3)
    for i in range(len(rows) - 2):
        b1 = rows[i]      # first candle (bearish)
        b2 = rows[i + 1]  # star (small body)
        b3 = rows[i + 2]  # third candle (bullish)

        o1, h1, l1, c1 = float(b1.Open), float(b1.High), float(b1.Low), float(b1.Close)
        o2, h2, l2, c2 = float(b2.Open), float(b2.High), float(b2.Low), float(b2.Close)
        o3, h3, l3, c3 = float(b3.Open), float(b3.High), float(b3.Low), float(b3.Close)

        rng1 = h1 - l1
        rng2 = h2 - l2
        rng3 = h3 - l3
        if rng1 <= 0 or rng3 <= 0:
            continue

        # Bar 1: bearish, body > 50% of range
        if not (c1 < o1 and abs(c1 - o1) / rng1 > 0.50):
            continue

        # Bar 2: small body < 30% of range
        if rng2 > 0 and abs(c2 - o2) / rng2 >= 0.30:
            continue

        # Bar 3: bullish, body > 40% of range
        if not (c3 > o3 and abs(c3 - o3) / rng3 > 0.40):
            continue

        # Bar 3 closes above midpoint of bar 1
        mid1 = (o1 + c1) / 2.0
        if c3 > mid1:
            return True

    return False
