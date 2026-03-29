"""Piercing Variant enhancer.

Practical cash-market adaptation of textbook Piercing Line for markets where
true gap-down opens are uncommon.

Two-candle bullish reversal pattern:
  bar[-2]: bearish candle with a meaningful body
  bar[-1]: bullish candle that opens near/below the prior close, then closes
           above the midpoint of bar[-2] body, but still below bar[-2] open
"""

from __future__ import annotations

import pandas as pd


def check(
    prices: pd.DataFrame,
    ticker: str,
    lookback: int = 2,
    prev_body_pct_min: float = 0.25,
    curr_body_pct_min: float = 0.20,
    open_above_prev_close_max_pct: float = 0.03,
) -> bool:
    """True if *ticker* shows a practical piercing-line variant in its latest 2 bars."""
    g = prices.loc[prices["Ticker"] == ticker].sort_values("Date").tail(max(lookback, 2))
    if len(g) < 2:
        return False

    prev, curr = list(g.tail(2).itertuples(index=False))
    o_prev, h_prev, l_prev, c_prev = float(prev.Open), float(prev.High), float(prev.Low), float(prev.Close)
    o_curr, h_curr, l_curr, c_curr = float(curr.Open), float(curr.High), float(curr.Low), float(curr.Close)
    rng_prev = h_prev - l_prev
    rng_curr = h_curr - l_curr
    if rng_prev <= 0 or rng_curr <= 0:
        return False

    prev_body = o_prev - c_prev
    curr_body = c_curr - o_curr
    if not (c_prev < o_prev and prev_body / rng_prev >= prev_body_pct_min):
        return False
    if not (c_curr > o_curr and curr_body / rng_curr >= curr_body_pct_min):
        return False

    mid_prev = (o_prev + c_prev) / 2.0
    return bool(
        o_curr <= o_prev
        and o_curr <= c_prev * (1.0 + open_above_prev_close_max_pct)
        and c_curr > mid_prev
        and c_curr < o_prev
    )