"""Pattern detection registry.

Each pattern module exports a ``detect()`` function with the signature:

    detect(prices: pd.DataFrame, *, as_of_date: pd.Timestamp, **params) -> pd.DataFrame

The returned DataFrame must include the standard columns listed in
``STANDARD_SIGNAL_COLS``.
"""

from __future__ import annotations

STANDARD_SIGNAL_COLS = [
    "signal_date",
    "ticker",
    "pattern",
    "pattern_family",
    "entry_price",
    "stop_pct",
    "stop_price",
    "score_trend",
    "score_setup",
    "score_volume",
    "score_rsi",
    "score_risk",
    "score_pattern",
    "sma50_slope_pct",
    "ma_slope_bonus",
    "pattern_bonus",
    "signal_score",
    "consensus_count",
]
