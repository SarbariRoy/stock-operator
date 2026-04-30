"""Parallel stop-risk modeling for calibrated reliability scores.

This module learns a separate monotonic stop-risk model, preserves the original
heuristic score in a companion column, and can apply a continuous stop-risk
penalty to the published ranking score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .penalties import compute_signal_penalty_features, get_recent_signal_lookback_days

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_STOP_RISK_MODEL_JSON = DATA_DIR / "st_lt_signal_stop_risk_model.json"
STOP_RISK_OUTPUT_COLUMNS = [
    "signal_stop_risk",
    "signal_stop_risk_5d",
    "signal_gap_through_stop_risk",
    "signal_mae_exceeds_stop_risk",
    "signal_reliability_score",
    "signal_score_pre_stop_risk_penalty",
    "score_penalty_stop_risk",
    "score_penalty_stop_risk_method",
    "score_penalty_stop_risk_gated",
]
DEFAULT_STOP_RISK_PENALTY_POLICY = {
    "enabled": True,
    "method": "continuous_power",
    "risk_floor": 0.25,
    "risk_full_penalty": 0.60,
    "max_penalty": 24.0,
    "power": 1.6,
    "hard_gate_enabled": False,
    "hard_gate_threshold": 0.80,
    "reliability_cap_enabled": True,
    "reliability_cap_threshold": 75.0,
    "reliability_cap_score_cap": 97.0,
}
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
    "feature_gap_sequence_risk",
    "feature_exhaustion_risk",
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


def get_default_stop_risk_penalty_policy() -> dict[str, object]:
    return dict(DEFAULT_STOP_RISK_PENALTY_POLICY)


def ensure_stop_risk_columns(signals_df: pd.DataFrame) -> pd.DataFrame:
    out = signals_df.copy()
    for column in STOP_RISK_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _resolve_stop_risk_penalty_policy(payload: dict | None) -> dict[str, object]:
    resolved = get_default_stop_risk_penalty_policy()
    raw_policy = payload.get("stop_risk_penalty_policy") if isinstance(payload, dict) else None
    if not isinstance(raw_policy, dict):
        return resolved

    bool_keys = {"enabled", "hard_gate_enabled", "reliability_cap_enabled"}
    float_keys = {
        "risk_floor",
        "risk_full_penalty",
        "max_penalty",
        "power",
        "hard_gate_threshold",
        "reliability_cap_threshold",
        "reliability_cap_score_cap",
    }
    for key, value in raw_policy.items():
        if key in bool_keys:
            resolved[key] = bool(value)
        elif key in float_keys:
            try:
                resolved[key] = float(value)
            except (TypeError, ValueError):
                continue
        elif key == "method":
            resolved[key] = str(value)

    resolved["risk_floor"] = max(0.0, min(1.0, float(resolved["risk_floor"])))
    resolved["risk_full_penalty"] = max(float(resolved["risk_floor"]) + 1e-6, min(1.0, float(resolved["risk_full_penalty"])))
    resolved["max_penalty"] = max(0.0, float(resolved["max_penalty"]))
    resolved["power"] = max(1.0, float(resolved["power"]))
    resolved["hard_gate_threshold"] = max(0.0, min(1.0, float(resolved["hard_gate_threshold"])))
    resolved["reliability_cap_threshold"] = max(0.0, min(100.0, float(resolved["reliability_cap_threshold"])))
    resolved["reliability_cap_score_cap"] = max(0.0, min(100.0, float(resolved["reliability_cap_score_cap"])))
    if str(resolved.get("method", "continuous_power")) != "continuous_power":
        resolved["method"] = "continuous_power"
    return resolved


def _resolve_base_signal_score(signals_df: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(signals_df.get("signal_score"), errors="coerce").fillna(0.0)
    pre_penalty = pd.to_numeric(signals_df.get("signal_score_pre_stop_risk_penalty"), errors="coerce")
    if isinstance(pre_penalty, pd.Series):
        return pre_penalty.fillna(score)
    return score


def _compute_stop_risk_penalty(stop_risk: pd.Series, policy: dict[str, object]) -> tuple[pd.Series, pd.Series]:
    risk = pd.to_numeric(stop_risk, errors="coerce").clip(lower=0.0, upper=1.0)
    risk_floor = float(policy.get("risk_floor", DEFAULT_STOP_RISK_PENALTY_POLICY["risk_floor"]))
    risk_full_penalty = float(policy.get("risk_full_penalty", DEFAULT_STOP_RISK_PENALTY_POLICY["risk_full_penalty"]))
    max_penalty = float(policy.get("max_penalty", DEFAULT_STOP_RISK_PENALTY_POLICY["max_penalty"]))
    power = float(policy.get("power", DEFAULT_STOP_RISK_PENALTY_POLICY["power"]))

    normalized = ((risk - risk_floor) / max(risk_full_penalty - risk_floor, 1e-6)).clip(lower=0.0, upper=1.0)
    penalty = (max_penalty * normalized.pow(power)).fillna(0.0)
    gate_threshold = float(policy.get("hard_gate_threshold", DEFAULT_STOP_RISK_PENALTY_POLICY["hard_gate_threshold"]))
    if bool(policy.get("hard_gate_enabled", False)):
        gated = risk > gate_threshold
    else:
        gated = pd.Series(False, index=risk.index, dtype="bool")
    return penalty, gated


def _apply_reliability_score_cap(
    signal_score: pd.Series,
    reliability_score: pd.Series,
    policy: dict[str, object],
) -> pd.Series:
    capped = pd.to_numeric(signal_score, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0).copy()
    if not bool(policy.get("reliability_cap_enabled", False)):
        return capped

    reliability = pd.to_numeric(reliability_score, errors="coerce")
    threshold = float(policy.get("reliability_cap_threshold", DEFAULT_STOP_RISK_PENALTY_POLICY["reliability_cap_threshold"]))
    score_cap = float(policy.get("reliability_cap_score_cap", DEFAULT_STOP_RISK_PENALTY_POLICY["reliability_cap_score_cap"]))
    mask = reliability < threshold
    if mask.any():
        capped.loc[mask] = np.minimum(capped.loc[mask], score_cap)
    return capped


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
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
    class_weight = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)).astype("float64")
    if sample_weight is None:
        combined_weight = class_weight
    else:
        combined_weight = class_weight * np.clip(np.asarray(sample_weight, dtype="float64"), a_min=0.0, a_max=None)
    weight_sum = max(1.0, float(combined_weight.sum()))

    for _ in range(int(max_iter)):
        logits = X @ weights + intercept
        probs = _sigmoid(logits)
        residual = (probs - y) * combined_weight
        grad_w = (X.T @ residual) / weight_sum
        grad_w += (float(l2) / max(1.0, float(n_samples))) * weights
        grad_b = float(residual.sum() / weight_sum)

        weights -= float(learning_rate) * grad_w
        intercept -= float(learning_rate) * grad_b

        if max(float(np.max(np.abs(grad_w))), abs(grad_b)) < 1e-6:
            break

    return weights, intercept


def _fit_isotonic_regression(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    if probabilities.size == 0:
        return [], []

    order = np.argsort(probabilities)
    xs = probabilities[order].astype("float64")
    ys = outcomes[order].astype("float64")
    weights = np.ones_like(xs, dtype="float64") if sample_weight is None else np.asarray(sample_weight, dtype="float64")[order]

    blocks: list[dict[str, float | int]] = []
    for idx, (x_val, y_val, weight_val) in enumerate(zip(xs, ys, weights)):
        block_weight = max(float(weight_val), 0.0)
        blocks.append(
            {
                "start": idx,
                "end": idx,
                "weight": block_weight,
                "sum_y": float(y_val) * block_weight,
                "value": float(y_val) if block_weight > 0 else 0.0,
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
    base_signal_score = _resolve_base_signal_score(out)
    out["signal_score_pre_stop_risk_penalty"] = base_signal_score.round(4)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), dict):
        out["score_penalty_stop_risk"] = 0.0
        out["score_penalty_stop_risk_method"] = pd.NA
        out["score_penalty_stop_risk_gated"] = False
        out["signal_score"] = base_signal_score.round(1)
        return out

    recent_signal_lookback_days = get_recent_signal_lookback_days(payload, default=int(payload.get("recent_signal_lookback_days", 20) if isinstance(payload.get("recent_signal_lookback_days"), (int, float)) else 20))
    model_input = out.copy()
    model_input["signal_score"] = base_signal_score
    featured = prepare_stop_risk_features(
        model_input,
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
        policy = _resolve_stop_risk_penalty_policy(payload)
        reliability = pd.to_numeric(out.get("signal_reliability_score"), errors="coerce")
        if bool(policy.get("enabled", False)):
            penalty, gated = _compute_stop_risk_penalty(stop_risk, policy)
            adjusted_score = (base_signal_score - penalty).clip(lower=0.0, upper=100.0)
            adjusted_score = adjusted_score.mask(gated, 0.0)
            adjusted_score = _apply_reliability_score_cap(adjusted_score, reliability, policy)
            out["score_penalty_stop_risk"] = penalty.round(4)
            out["score_penalty_stop_risk_method"] = str(policy.get("method", "continuous_power"))
            out["score_penalty_stop_risk_gated"] = gated.astype(bool)
            out["signal_score"] = adjusted_score.round(1)
        else:
            out["score_penalty_stop_risk"] = 0.0
            out["score_penalty_stop_risk_method"] = pd.NA
            out["score_penalty_stop_risk_gated"] = False
            out["signal_score"] = _apply_reliability_score_cap(base_signal_score, reliability, policy).round(1)
    return out
