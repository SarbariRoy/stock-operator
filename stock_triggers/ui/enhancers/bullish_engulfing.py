"""Bullish Engulfing enhancer.

Two-candle pattern:
  bar[-2]: red candle (close < open)
  bar[-1]: green candle (close > open) whose real body fully engulfs bar[-2]'s
           real body  (open[-1] <= close[-2] and close[-1] >= open[-2])
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
) -> bool:
    """True if *ticker* shows a bullish engulfing in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 2))
    if len(g) < 2:
        return False

    rows = list(g.itertuples(index=False))
    for i in range(len(rows) - 1):
        prev = rows[i]
        curr = rows[i + 1]

        o_prev, c_prev = float(prev.Open), float(prev.Close)
        o_curr, c_curr = float(curr.Open), float(curr.Close)

        # Previous candle must be red (bearish)
        if not c_prev < o_prev:
            continue

        # Current candle must be green (bullish)
        if not c_curr > o_curr:
            continue

        # Current body engulfs previous body
        if o_curr <= c_prev and c_curr >= o_prev:
            return True

    return False
