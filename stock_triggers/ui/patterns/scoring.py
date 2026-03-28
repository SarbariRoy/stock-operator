"""Shared scoring utilities for all signal patterns."""

from __future__ import annotations

import pandas as pd


def clip_score(value: float) -> float:
    """Clamp *value* to [0, 100]."""
    return max(0.0, min(100.0, float(value)))


# Component weights for signal_score.
WEIGHT_TREND = 0.28
WEIGHT_SETUP = 0.28
WEIGHT_VOLUME = 0.19
WEIGHT_RISK = 0.20
WEIGHT_RSI = 0.05


def build_score_components(
    *,
    trend_strength_pct: float,
    setup_strength_pct: float,
    volume_ratio: float,
    stop_pct_eff: float,
    rsi_value: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return (score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score)."""
    score_trend = clip_score(50.0 + trend_strength_pct * 5.0)
    score_setup = clip_score(50.0 + setup_strength_pct * 8.0)
    score_volume = clip_score(40.0 + volume_ratio * 20.0)
    score_risk = clip_score(100.0 - stop_pct_eff * 6.0)

    if rsi_value is None or pd.isna(rsi_value):
        score_rsi = 50.0
    else:
        score_rsi = clip_score(rsi_value)

    signal_score = round(
        (WEIGHT_TREND * score_trend)
        + (WEIGHT_SETUP * score_setup)
        + (WEIGHT_VOLUME * score_volume)
        + (WEIGHT_RISK * score_risk)
        + (WEIGHT_RSI * score_rsi),
        1,
    )
    return (
        round(score_trend, 1),
        round(score_setup, 1),
        round(score_volume, 1),
        round(score_risk, 1),
        round(score_rsi, 1),
        signal_score,
    )
