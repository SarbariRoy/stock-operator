from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .markov import apply_signal_markov_model
from .penalties import apply_signal_penalty_weights, compute_signal_penalty_features, get_recent_signal_lookback_days
from .scoring import apply_pattern_family_bonus
from .stop_risk import apply_signal_stop_risk_model


def load_existing_signal_history(path: Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Signal history file not found: {path}")
    signals = pd.read_csv(path)
    for column in required_columns:
        if column not in signals.columns:
            signals[column] = pd.NA
    return signals


def rescore_signal_history(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    pattern_weights: dict | None,
    penalty_payload: dict | None,
    markov_payload: dict | None,
    markov_mode: str = "auto",
    stop_risk_payload: dict | None,
) -> pd.DataFrame:
    rescored = apply_pattern_family_bonus(signals_df, pattern_weights)
    rescored = compute_signal_penalty_features(
        rescored,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=get_recent_signal_lookback_days(penalty_payload),
    )
    rescored = apply_signal_penalty_weights(rescored, penalty_payload)
    rescored = apply_signal_markov_model(
        rescored,
        prices_df,
        markov_payload,
        markov_mode=markov_mode,
    )
    rescored = apply_signal_stop_risk_model(
        rescored,
        prices_df,
        stop_risk_payload,
        breakout_days=int(breakout_days),
    )
    return rescored