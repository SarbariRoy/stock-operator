"""Bullish Harami enhancer.

Two-candle bullish reversal pattern:
  bar[-2]: bearish candle with a reasonably sized real body
  bar[-1]: bullish candle whose real body sits inside bar[-2]'s real body
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
) -> bool:
    """True if *ticker* shows a bullish harami in its last *lookback* bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 2))
    if len(g) < 2:
        return False

    rows = list(g.itertuples(index=False))
    for i in range(len(rows) - 1):
        prev = rows[i]
        curr = rows[i + 1]

        o_prev, h_prev, l_prev, c_prev = float(prev.Open), float(prev.High), float(prev.Low), float(prev.Close)
        o_curr, c_curr = float(curr.Open), float(curr.Close)
        rng_prev = h_prev - l_prev
        if rng_prev <= 0:
            continue

        prev_body = abs(c_prev - o_prev)
        curr_body = abs(c_curr - o_curr)

        if not (c_prev < o_prev and prev_body / rng_prev >= 0.40):
            continue
        if not (c_curr > o_curr):
            continue
        if curr_body <= 0:
            continue

        prev_body_low = min(o_prev, c_prev)
        prev_body_high = max(o_prev, c_prev)
        curr_body_low = min(o_curr, c_curr)
        curr_body_high = max(o_curr, c_curr)

        if curr_body_low >= prev_body_low and curr_body_high <= prev_body_high:
            return True

    return False
