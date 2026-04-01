"""Parallel stop-risk modeling for calibrated reliability scores.

This module intentionally leaves the existing heuristic signal_score path intact.
It learns and applies a separate monotonic stop-risk model whose outputs can be
used for ranking or downstream filtering without changing the base setup score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .penalties import compute_signal_penalty_features, get_recent_signal_lookback_days

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_STOP_RISK_MODEL_JSON = DATA_DIR / "signal_stop_risk_model.json"
STOP_RISK_OUTPUT_COLUMNS = [
    "signal_stop_risk",
    "signal_stop_risk_5d",
    "signal_gap_through_stop_risk",
    "signal_mae_exceeds_stop_risk",
    "signal_reliability_score",
]
STOP_RISK_SCORE_COMPONENT_FEATURES = [
    "signal_score",
    "score_trend",
    "score_setup",
    "score_volume",
    "score_rsi",
    "score_risk",
    "score_pattern",
]
STOP_RISK_BONUS_FEATURES = [
    "ma_slope_bonus",
    "pattern_bonus",
    "score_penalty_total",
    "consensus_count",
]
STOP_RISK_ROW_CONTEXT_FEATURES = [
    "feature_recent_signal_count",
    "feature_close_vs_prev_high_pct",
    "feature_close_vs_sma50_pct",
    "feature_gap_pct",
    "feature_range_vs_atr",
]
STOP_RISK_REGIME_FEATURES = [
    "regime_pct_above_sma50",
    "regime_pct_above_sma200",
    "regime_median_ret_20d_pct",
    "regime_median_close_vs_sma50_pct",
]
STOP_RISK_NUMERIC_FEATURES = [
    *STOP_RISK_SCORE_COMPONENT_FEATURES,
    *STOP_RISK_BONUS_FEATURES,
    *STOP_RISK_ROW_CONTEXT_FEATURES,
    *STOP_RISK_REGIME_FEATURES,
]
STOP_RISK_TARGETS = {
    "stop_before_target": "signal_stop_risk",
    "stop_within_5d": "signal_stop_risk_5d",
    "gap_through_stop": "signal_gap_through_stop_risk",
    "mae_exceeds_stop": "signal_mae_exceeds_stop_risk",
}
STOP_RISK_FAMILY_LEVELS = ["A", "B", "C", "D", "E", "F", "G"]
STOP_RISK_FEATURE_SET_PRESETS = {
    "full": list(STOP_RISK_NUMERIC_FEATURES),
    "scores_only": [
        *STOP_RISK_SCORE_COMPONENT_FEATURES,
        *STOP_RISK_BONUS_FEATURES,
    ],
    "scores_plus_row_context": [
        *STOP_RISK_SCORE_COMPONENT_FEATURES,
        *STOP_RISK_BONUS_FEATURES,
        *STOP_RISK_ROW_CONTEXT_FEATURES,
    ],
    "scores_plus_regime": [
        *STOP_RISK_SCORE_COMPONENT_FEATURES,
        *STOP_RISK_BONUS_FEATURES,
        *STOP_RISK_REGIME_FEATURES,
    ],
}


def load_signal_stop_risk_model(path: Path = DEFAULT_SIGNAL_STOP_RISK_MODEL_JSON) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def ensure_stop_risk_columns(signals_df: pd.DataFrame) -> pd.DataFrame:
    out = signals_df.copy()
    for column in STOP_RISK_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    max_iter: int = 2000,
    l2: float = 0.5,
) -> tuple[np.ndarray, float]:
    if X.size == 0:
        return np.zeros(0, dtype="float64"), 0.0

    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype="float64")
    intercept = 0.0

    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)).astype("float64")
    weight_sum = max(1.0, float(sample_weight.sum()))

    for _ in range(int(max_iter)):
        logits = X @ weights + intercept
        probs = _sigmoid(logits)
        residual = (probs - y) * sample_weight
        grad_w = (X.T @ residual) / weight_sum
        grad_w += (float(l2) / max(1.0, float(n_samples))) * weights
        grad_b = float(residual.sum() / weight_sum)

        weights -= float(learning_rate) * grad_w
        intercept -= float(learning_rate) * grad_b

        if max(float(np.max(np.abs(grad_w))), abs(grad_b)) < 1e-6:
            break

    return weights, intercept


def _fit_isotonic_regression(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[list[float], list[float]]:
    if probabilities.size == 0:
        return [], []

    order = np.argsort(probabilities)
    xs = probabilities[order].astype("float64")
    ys = outcomes[order].astype("float64")

    blocks: list[dict[str, float | int]] = []
    for idx, (x_val, y_val) in enumerate(zip(xs, ys)):
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": 1.0,
                "sum_y": float(y_val),
                "value": float(y_val),
                "upper": float(x_val),
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_sum_y = float(left["sum_y"]) + float(right["sum_y"])
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "sum_y": merged_sum_y,
                    "value": merged_sum_y / merged_weight if merged_weight > 0 else 0.0,
                    "upper": float(right["upper"]),
                }
            )

    return [round(float(block["upper"]), 10) for block in blocks], [round(float(block["value"]), 10) for block in blocks]


def _apply_isotonic_regression(probabilities: np.ndarray, upper_bounds: list[float], values: list[float]) -> np.ndarray:
    if probabilities.size == 0:
        return probabilities.astype("float64")
    if not upper_bounds or not values:
        return probabilities.astype("float64")

    bounds = np.asarray(upper_bounds, dtype="float64")
    calibrated_values = np.asarray(values, dtype="float64")
    indices = np.searchsorted(bounds, probabilities.astype("float64"), side="left")
    indices = np.clip(indices, 0, len(calibrated_values) - 1)
    return calibrated_values[indices]


def _prepare_market_regime_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame(columns=[
            "signal_date",
            "regime_pct_above_sma50",
            "regime_pct_above_sma200",
            "regime_median_ret_20d_pct",
            "regime_median_close_vs_sma50_pct",
        ])

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices.sort_values(["Ticker", "Date"], inplace=True)
    grouped = prices.groupby("Ticker", sort=False)
    prices["SMA50"] = grouped["Close"].transform(lambda s: s.rolling(50).mean())
    prices["SMA200"] = grouped["Close"].transform(lambda s: s.rolling(200).mean())
    prices["Ret20dPct"] = grouped["Close"].transform(lambda s: ((s / s.shift(20)) - 1.0) * 100.0)
    prices["CloseVsSMA50Pct"] = ((prices["Close"] / prices["SMA50"]) - 1.0) * 100.0
    prices["AboveSMA50"] = (prices["Close"] > prices["SMA50"]).astype("float64")
    prices["AboveSMA200"] = (prices["Close"] > prices["SMA200"]).astype("float64")

    regime = (
        prices.groupby("Date", sort=True)
        .agg(
            regime_pct_above_sma50=("AboveSMA50", "mean"),
            regime_pct_above_sma200=("AboveSMA200", "mean"),
            regime_median_ret_20d_pct=("Ret20dPct", "median"),
            regime_median_close_vs_sma50_pct=("CloseVsSMA50Pct", "median"),
        )
        .reset_index()
    )
    regime["signal_date"] = pd.to_datetime(regime["Date"], errors="coerce").dt.date.astype("string")
    regime.drop(columns=["Date"], inplace=True)
    regime["regime_pct_above_sma50"] = (pd.to_numeric(regime["regime_pct_above_sma50"], errors="coerce") * 100.0).round(4)
    regime["regime_pct_above_sma200"] = (pd.to_numeric(regime["regime_pct_above_sma200"], errors="coerce") * 100.0).round(4)
    return regime


def prepare_stop_risk_features(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    recent_signal_lookback_days: int,
) -> pd.DataFrame:
    featured = compute_signal_penalty_features(
        signals_df,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(recent_signal_lookback_days),
    )
    featured = ensure_stop_risk_columns(featured)
    featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")

    regime = _prepare_market_regime_features(prices_df)
    if not regime.empty:
        featured = featured.merge(regime, on="signal_date", how="left")
    else:
        for column in STOP_RISK_NUMERIC_FEATURES:
            if column.startswith("regime_") and column not in featured.columns:
                featured[column] = pd.NA

    return featured


def _build_feature_frame(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    numeric_features = list(model_payload.get("numeric_features", []))
    impute_medians = model_payload.get("impute_medians", {}) if isinstance(model_payload.get("impute_medians"), dict) else {}
    scaler_means = model_payload.get("scaler_means", {}) if isinstance(model_payload.get("scaler_means"), dict) else {}
    scaler_stds = model_payload.get("scaler_stds", {}) if isinstance(model_payload.get("scaler_stds"), dict) else {}
    family_levels = list(model_payload.get("family_levels", STOP_RISK_FAMILY_LEVELS))

    parts: list[np.ndarray] = []
    for feature_name in numeric_features:
        series = pd.to_numeric(feature_df.get(feature_name), errors="coerce")
        median_value = float(impute_medians.get(feature_name, 0.0))
        mean_value = float(scaler_means.get(feature_name, 0.0))
        std_value = float(scaler_stds.get(feature_name, 1.0)) or 1.0
        filled = series.fillna(median_value).astype("float64")
        parts.append((((filled - mean_value) / std_value).to_numpy()).reshape(-1, 1))

    families = feature_df.get("pattern_family", pd.Series("", index=feature_df.index)).astype(str).str.strip().str.upper()
    for family in family_levels:
        parts.append((families == family).astype("float64").to_numpy().reshape(-1, 1))

    if not parts:
        return np.zeros((len(feature_df), 0), dtype="float64")
    return np.hstack(parts).astype("float64")


def _predict_target_probabilities(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    if feature_df.empty:
        return np.zeros(0, dtype="float64")

    X = _build_feature_frame(feature_df, model_payload)
    coefficients = np.asarray(model_payload.get("coefficients", []), dtype="float64")
    intercept = float(model_payload.get("intercept", 0.0))
    raw_probabilities = _sigmoid((X @ coefficients) + intercept)
    upper_bounds = list(model_payload.get("isotonic_upper_bounds", []))
    values = list(model_payload.get("isotonic_values", []))
    return np.clip(_apply_isotonic_regression(raw_probabilities, upper_bounds, values), 0.0, 1.0)


def apply_signal_stop_risk_model(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    payload: dict | None,
    *,
    breakout_days: int = 40,
) -> pd.DataFrame:
    out = ensure_stop_risk_columns(signals_df)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), dict):
        return out

    recent_signal_lookback_days = get_recent_signal_lookback_days(payload, default=int(payload.get("recent_signal_lookback_days", 20) if isinstance(payload.get("recent_signal_lookback_days"), (int, float)) else 20))
    featured = prepare_stop_risk_features(
        out,
        prices_df,
        breakout_days=int(payload.get("breakout_days", breakout_days)),
        recent_signal_lookback_days=int(recent_signal_lookback_days),
    )

    for target_name, output_column in STOP_RISK_TARGETS.items():
        target_payload = payload.get("targets", {}).get(target_name)
        if not isinstance(target_payload, dict):
            continue
        probabilities = _predict_target_probabilities(featured, target_payload)
        out[output_column] = pd.Series(np.round(probabilities, 4), index=out.index)

    stop_risk = pd.to_numeric(out.get("signal_stop_risk"), errors="coerce")
    if isinstance(stop_risk, pd.Series):
        out["signal_reliability_score"] = ((1.0 - stop_risk.clip(lower=0.0, upper=1.0)) * 100.0).round().astype("Int64")
    return out
