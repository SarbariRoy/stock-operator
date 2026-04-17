from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def parse_optional_date(value: str, *, arg_name: str) -> pd.Timestamp | None:
    if not str(value or "").strip():
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise SystemExit(f"Invalid {arg_name}: {value}")
    return pd.Timestamp(parsed)


def filter_by_date_window(
    df: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or (start_date is None and end_date is None):
        return df.copy()

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = parsed_dates.notna()
    if start_date is not None:
        mask &= parsed_dates >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= parsed_dates <= pd.Timestamp(end_date)
    return df.loc[mask].copy()


def compute_recency_weights(
    signal_dates: Iterable[object] | pd.Series,
    *,
    half_life_months: float,
    reference_date: object | None = None,
) -> pd.Series:
    dates = pd.to_datetime(pd.Series(signal_dates), errors="coerce")
    weights = pd.Series(1.0, index=dates.index, dtype="float64")
    if float(half_life_months) <= 0:
        return weights

    valid_dates = dates.dropna()
    if valid_dates.empty:
        return weights

    ref = pd.to_datetime(reference_date, errors="coerce") if reference_date is not None else valid_dates.max()
    if pd.isna(ref):
        ref = valid_dates.max()

    age_days = (pd.Timestamp(ref) - dates).dt.days.clip(lower=0)
    age_months = age_days / 30.4375
    decay = np.log(2.0) / float(half_life_months)
    valid_mask = age_months.notna()
    weights.loc[valid_mask] = np.exp(-decay * age_months.loc[valid_mask].astype("float64"))
    return weights.astype("float64")


def add_recency_weights(
    df: pd.DataFrame,
    *,
    date_col: str = "signal_date",
    half_life_months: float = 0.0,
    reference_date: object | None = None,
    weight_col: str = "sample_weight",
) -> pd.DataFrame:
    out = df.copy()
    out[weight_col] = compute_recency_weights(
        out.get(date_col, pd.Series(index=out.index, dtype="object")),
        half_life_months=float(half_life_months),
        reference_date=reference_date,
    ).to_numpy()
    return out


def get_sample_weight_series(df: pd.DataFrame, *, weight_col: str = "sample_weight") -> pd.Series:
    if weight_col in df.columns:
        weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        return weights.astype("float64")
    return pd.Series(1.0, index=df.index, dtype="float64")


def weighted_mean(values: Iterable[object] | pd.Series, weights: Iterable[object] | pd.Series | None = None) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if weights is None:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else 0.0

    weight_series = pd.to_numeric(pd.Series(weights), errors="coerce")
    valid_mask = series.notna() & weight_series.notna() & (weight_series > 0)
    if not valid_mask.any():
        return 0.0
    valid_values = series.loc[valid_mask].astype("float64")
    valid_weights = weight_series.loc[valid_mask].astype("float64")
    total_weight = float(valid_weights.sum())
    if total_weight <= 0:
        return 0.0
    return float(np.average(valid_values, weights=valid_weights))