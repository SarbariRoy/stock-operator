"""Candle-shape enhancer registry.

Each enhancer module exports a ``check()`` function:

    check(prices: pd.DataFrame, ticker: str, lookback: int = 2) -> bool

Returns True when the candle shape is detected in the last *lookback* bars
for the given ticker.
"""

from __future__ import annotations
