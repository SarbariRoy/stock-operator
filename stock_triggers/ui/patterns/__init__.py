"""Pattern detection registry.

Each pattern module exports a ``detect()`` function with the signature:

    detect(prices: pd.DataFrame, *, as_of_date: pd.Timestamp, **params) -> pd.DataFrame

The returned DataFrame must include the standard columns listed in
``STANDARD_SIGNAL_COLS``.
"""

from __future__ import annotations

FEATURE_SIGNAL_COLS = [
    "feature_recent_signal_count",
    "feature_close_vs_prev_high_pct",
    "feature_close_vs_sma50_pct",
    "feature_gap_pct",
    "feature_range_vs_atr",
    "feature_gap_sequence_risk",
    "feature_exhaustion_risk",
]

PENALTY_SIGNAL_COLS = [
    "score_penalty_crowding",
    "score_penalty_extension",
    "score_penalty_gap_shock",
    "score_penalty_total",
    "score_penalty_stop_risk",
]

STOP_RISK_SIGNAL_COLS = [
    "signal_stop_risk",
    "signal_stop_risk_5d",
    "signal_gap_through_stop_risk",
    "signal_mae_exceeds_stop_risk",
    "signal_reliability_score",
    "signal_score_pre_stop_risk_penalty",
    "score_penalty_stop_risk_method",
    "score_penalty_stop_risk_gated",
]

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
    *FEATURE_SIGNAL_COLS,
    "pattern_bonus",
    *PENALTY_SIGNAL_COLS,
    "signal_score",
    *STOP_RISK_SIGNAL_COLS,
    "consensus_count",
]
