"""Apply trained ST score model to signals to generate st_score predictions.

The ST (Short-Term) score predicts the probability of hitting a 3% target within
7 days before hitting a 3% stop loss, expressed as a 0-100 score. This module
applies a pre-trained logistic regression model with isotonic calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .markov import build_markov_state_table, load_signal_markov_model
from .stop_risk import _sigmoid, _apply_isotonic_regression

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_ST_SCORE_MODEL_JSON = DATA_DIR / "signal_st_score_model.json"

ST_OUTPUT_COLUMNS = ["st_score", "st_score_pre_model", "markov_state_encoded"]
ST_RANK_BLEND_WEIGHT = 0.25

ST_NUMERIC_FEATURES = [
    "score_trend",
    "score_setup",
    "score_volume",
    "score_rsi",
    "score_risk",
    "markov_p_continuation",
    "markov_p_adverse",
    "gap_pct",
    "volatility_pct",
    "momentum_1d_pct",
    "momentum_3d_pct",
    "momentum_5d_pct",
    "consensus_count",
    "feature_recent_signal_count",
]

ST_FAMILY_LEVELS = ["A", "B", "C", "D", "E", "F", "G"]

# Markov state constants
MARKOV_STATE_LEVELS = (
    "constructive_trend",
    "fresh_breakout",
    "extended_breakout",
    "sideways",
    "breakdown_risk",
)

MARKOV_CONTINUATION_STATES = {"constructive_trend", "fresh_breakout"}
MARKOV_ADVERSE_STATES = {"extended_breakout", "breakdown_risk"}


def _markov_probs_from_state(state_name: str, transitions: dict, continuation_states: set[str], adverse_states: set[str]) -> tuple[float, float]:
    row = transitions.get(str(state_name), {}) if isinstance(transitions, dict) else {}
    if not isinstance(row, dict):
        return 0.0, 0.0
    p_cont = 0.0
    p_adv = 0.0
    for next_state, raw_prob in row.items():
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            continue
        if str(next_state) in continuation_states:
            p_cont += prob
        if str(next_state) in adverse_states:
            p_adv += prob
    return float(max(0.0, min(1.0, p_cont))), float(max(0.0, min(1.0, p_adv)))


def _ensure_markov_probabilities(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    out = signals_df.copy()
    if out.empty:
        return out

    # Ensure columns exist.
    if "markov_state" not in out.columns:
        out["markov_state"] = ""
    if "markov_p_continuation" not in out.columns:
        out["markov_p_continuation"] = 0.0
    if "markov_p_adverse" not in out.columns:
        out["markov_p_adverse"] = 0.0

    # Fill missing markov_state from price-derived state table.
    if not prices_df.empty:
        state_table = build_markov_state_table(prices_df)
        if not state_table.empty:
            merge_df = out.copy()
            merge_df["ticker"] = merge_df.get("ticker", pd.Series("", index=merge_df.index)).astype(str).str.strip().str.upper()
            merge_df["signal_date"] = pd.to_datetime(merge_df.get("signal_date"), errors="coerce").dt.date.astype("string")
            merge_df = merge_df.merge(state_table, on=["ticker", "signal_date"], how="left", suffixes=("", "_new"))
            if "markov_state_new" in merge_df.columns:
                merge_df["markov_state"] = merge_df["markov_state_new"].combine_first(merge_df.get("markov_state"))
                merge_df.drop(columns=["markov_state_new"], inplace=True)
            out["markov_state"] = merge_df.get("markov_state", out["markov_state"]).fillna("")

    markov_payload = load_signal_markov_model()
    transitions = markov_payload.get("transitions") if isinstance(markov_payload.get("transitions"), dict) else {}
    score_policy = markov_payload.get("score_policy") if isinstance(markov_payload.get("score_policy"), dict) else {}
    continuation_states = set(str(s) for s in score_policy.get("continuation_states", list(MARKOV_CONTINUATION_STATES)))
    adverse_states = set(str(s) for s in score_policy.get("adverse_states", list(MARKOV_ADVERSE_STATES)))

    if not transitions:
        return out

    p_cont_list: list[float] = []
    p_adv_list: list[float] = []
    for state in out.get("markov_state", pd.Series("", index=out.index)).fillna(""):
        p_cont, p_adv = _markov_probs_from_state(str(state), transitions, continuation_states, adverse_states)
        p_cont_list.append(round(p_cont, 4))
        p_adv_list.append(round(p_adv, 4))

    existing_cont = pd.to_numeric(out.get("markov_p_continuation"), errors="coerce").fillna(0.0)
    existing_adv = pd.to_numeric(out.get("markov_p_adverse"), errors="coerce").fillna(0.0)
    calc_cont = pd.Series(p_cont_list, index=out.index, dtype="float64")
    calc_adv = pd.Series(p_adv_list, index=out.index, dtype="float64")

    out["markov_p_continuation"] = np.where(existing_cont.abs() > 1e-12, existing_cont, calc_cont)
    out["markov_p_adverse"] = np.where(existing_adv.abs() > 1e-12, existing_adv, calc_adv)
    return out


def load_signal_st_score_model(path: Path = DEFAULT_ST_SCORE_MODEL_JSON) -> dict:
    """Load the ST score model from disk.
    
    Args:
        path: Path to the JSON model file.
        
    Returns:
        Model dict with keys: model, numeric_features, family_levels, etc.
        Returns empty dict if not found or invalid.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        # Model structure: { "model": {...}, "numeric_features": [...], ...}
        # or just the model dict itself
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def ensure_st_score_columns(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ST score output columns exist in DataFrame.
    
    Initializes missing columns with pd.NA to avoid KeyError during population.
    
    Args:
        signals_df: Input signals DataFrame.
        
    Returns:
        DataFrame with all ST output columns present (filled with pd.NA if missing).
    """
    out = signals_df.copy()
    for column in ST_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def compute_st_intraday_features(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday gap and volatility for each signal date.
    
    For each signal, looks up the price bar on signal_date and computes:
    - gap_pct: (close - open) / open * 100
    - volatility_pct: (high - low) / close * 100
    
    Args:
        signals_df: Signals with ticker, signal_date, entry_price.
        prices_df: Price history with Ticker, Date, Open, High, Low, Close.
        
    Returns:
        DataFrame with added gap_pct and volatility_pct columns (0 if missing data).
    """
    out = signals_df.copy()
    out["gap_pct"] = 0.0
    out["volatility_pct"] = 0.0
    
    if prices_df.empty:
        return out
    
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip().str.upper()
    prices = prices.dropna(subset=["Date"]).sort_values(["Ticker", "Date"])
    
    # Build price lookup: {(ticker, date): {Open, High, Low, Close}}
    price_lookup: dict[tuple[str, str], dict] = {}
    for _, row in prices.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        date_val = pd.to_datetime(row.get("Date"), errors="coerce")
        if pd.isna(date_val):
            continue
        date_str = date_val.date().isoformat()
        key = (ticker, date_str)
        price_lookup[key] = {
            "open": pd.to_numeric(row.get("Open"), errors="coerce"),
            "high": pd.to_numeric(row.get("High"), errors="coerce"),
            "low": pd.to_numeric(row.get("Low"), errors="coerce"),
            "close": pd.to_numeric(row.get("Close"), errors="coerce"),
        }
    
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        signal_date_raw = row.get("signal_date")
        if isinstance(signal_date_raw, str):
            date_str = signal_date_raw
        else:
            signal_date = pd.to_datetime(signal_date_raw, errors="coerce")
            if pd.isna(signal_date):
                continue
            date_str = signal_date.date().isoformat()
        
        key = (ticker, date_str)
        if key not in price_lookup:
            continue
        
        bar = price_lookup[key]
        open_price = bar.get("open")
        close_price = bar.get("close")
        high_price = bar.get("high")
        low_price = bar.get("low")
        
        # Compute gap_pct
        if pd.notna(open_price) and float(open_price) > 0 and pd.notna(close_price):
            gap_pct = ((float(close_price) - float(open_price)) / float(open_price)) * 100.0
            out.at[idx, "gap_pct"] = round(float(gap_pct), 4)
        
        # Compute volatility_pct
        if pd.notna(close_price) and float(close_price) > 0 and pd.notna(high_price) and pd.notna(low_price):
            volatility_pct = ((float(high_price) - float(low_price)) / float(close_price)) * 100.0
            out.at[idx, "volatility_pct"] = round(float(volatility_pct), 4)
    
    return out


def encode_markov_state(markov_state_str: str | None) -> dict[str, int]:
    """One-hot encode a markov state string to dict of encoded indicators.
    
    Example: "fresh_breakout" -> {
        "constructive_trend_encoded": 0,
        "fresh_breakout_encoded": 1,
        "extended_breakout_encoded": 0,
        "sideways_encoded": 0,
        "breakdown_risk_encoded": 0,
    }
    
    Args:
        markov_state_str: Markov state value (e.g., "fresh_breakout").
        
    Returns:
        Dict mapping {state}_encoded keys to 0 or 1 values.
    """
    state = str(markov_state_str).strip().lower() if markov_state_str else ""
    encoded = {}
    for level in MARKOV_STATE_LEVELS:
        encoded[f"{level}_encoded"] = 1 if state == level else 0
    return encoded


def _build_st_feature_frame(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    """Build feature matrix for logistic regression prediction.
    
    Replicates the training procedure:
    1. Select numeric features and impute with medians
    2. Standardize using model's scaler means/stds
    3. One-hot encode pattern_family
    4. Stack into feature matrix
    
    Args:
        feature_df: DataFrame with all required features.
        model_payload: Model parameters from signal_st_score_model.json.
        
    Returns:
        Feature matrix as np.ndarray of shape (n_rows, n_features).
    """
    numeric_features = list(model_payload.get("numeric_features", []))
    impute_medians = model_payload.get("impute_medians", {})
    if not isinstance(impute_medians, dict):
        impute_medians = {}
    
    scaler_means = model_payload.get("scaler_means", {})
    if not isinstance(scaler_means, dict):
        scaler_means = {}
    
    scaler_stds = model_payload.get("scaler_stds", {})
    if not isinstance(scaler_stds, dict):
        scaler_stds = {}
    
    family_levels = list(model_payload.get("family_levels", ST_FAMILY_LEVELS))
    
    parts: list[np.ndarray] = []
    
    # Normalize numeric features
    for feature_name in numeric_features:
        raw = feature_df[feature_name] if feature_name in feature_df.columns else pd.Series(0.0, index=feature_df.index)
        series = pd.to_numeric(raw, errors="coerce")
        median_value = float(impute_medians.get(feature_name, 0.0))
        mean_value = float(scaler_means.get(feature_name, 0.0))
        std_value = float(scaler_stds.get(feature_name, 1.0)) or 1.0
        filled = series.fillna(median_value).astype("float64")
        normalized = ((filled - mean_value) / std_value).to_numpy().reshape(-1, 1)
        parts.append(normalized)
    
    # One-hot encode pattern_family
    families = feature_df.get("pattern_family", pd.Series("", index=feature_df.index)).astype(str).str.strip().str.upper()
    for family in family_levels:
        parts.append((families == family).astype("float64").to_numpy().reshape(-1, 1))
    
    if not parts:
        return np.zeros((len(feature_df), 0), dtype="float64")
    
    return np.hstack(parts).astype("float64")


def _apply_isotonic_calibration(
    probabilities: np.ndarray,
    upper_bounds: list[float],
    values: list[float],
) -> np.ndarray:
    """Apply isotonic regression calibration to raw probabilities.
    
    Maps raw predicted probabilities through the fitted isotonic function
    to produce calibrated probability estimates.
    
    Args:
        probabilities: Raw probabilities from logistic model in [0, 1].
        upper_bounds: Isotonic upper bounds from training.
        values: Isotonic calibrated values from training.
        
    Returns:
        Calibrated probabilities clipped to [0, 1].
    """
    return np.clip(_apply_isotonic_regression(probabilities, upper_bounds, values), 0.0, 1.0)


def _predict_st_score_probabilities(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    """Predict P(hit 3% target within 7 days) for each signal.
    
    Applies logistic regression:
    1. Build feature frame from numeric features + pattern family
    2. Compute logit = X @ coefficients + intercept
    3. Apply sigmoid to get raw probabilities
    4. Apply isotonic regression calibration
    
    Args:
        feature_df: DataFrame with all required feature columns.
        model_payload: Model parameters from signal_st_score_model.json.
        
    Returns:
        Array of probabilities in [0, 1] for each row.
    """
    if feature_df.empty:
        return np.zeros(0, dtype="float64")
    
    X = _build_st_feature_frame(feature_df, model_payload)
    coefficients = np.asarray(model_payload.get("coefficients", []), dtype="float64")
    intercept = float(model_payload.get("intercept", 0.0))
    
    # Compute logit and apply sigmoid
    raw_probabilities = _sigmoid((X @ coefficients) + intercept)
    
    # Apply isotonic calibration
    upper_bounds = list(model_payload.get("isotonic_upper_bounds", []))
    values = list(model_payload.get("isotonic_values", []))
    
    return _apply_isotonic_calibration(raw_probabilities, upper_bounds, values)


def apply_st_score_model(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    payload: dict | None,
) -> pd.DataFrame:
    """Main entry point to apply ST score model to signals.
    
    Orchestrates feature preparation and model prediction:
    1. Ensure output columns exist
    2. Compute intraday gap/volatility features
    3. Load markov state and encode markov probabilities
    4. Merge all features with signals
    5. Predict probabilities using trained model
    6. Convert to st_score (0-100)
    
    Args:
        signals_df: Input signals with required columns (ticker, signal_date, pattern_family, etc.).
        prices_df: Price history with Date, Ticker, Open, High, Low, Close.
        payload: Model payload dict with keys: model (containing coefficients, etc), 
                 or None to skip scoring.
                 
    Returns:
        DataFrame with added st_score, st_score_pre_model, and markov_state_encoded columns.
        On error or missing model, st_score is pd.NA.
    """
    out = ensure_st_score_columns(signals_df)
    
    # If no payload or missing model, return with NA st_score
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        return out
    
    model_payload = payload.get("model", {})
    if not model_payload:
        return out
    
    # Compute intraday features
    featured = compute_st_intraday_features(out, prices_df)
    
    # Initialize markov encoding columns (one for each state)
    for level in MARKOV_STATE_LEVELS:
        col_name = f"{level}_encoded"
        if col_name not in featured.columns:
            featured[col_name] = 0
    
    # Encode markov state if available
    if "markov_state" in featured.columns:
        for idx, row in featured.iterrows():
            markov_state = str(row.get("markov_state", "")).strip()
            encoded = encode_markov_state(markov_state)
            for col_name, value in encoded.items():
                featured.at[idx, col_name] = int(value)
    
    # Ensure markov probability features are present and populated.
    featured = _ensure_markov_probabilities(featured, prices_df)
    
    # Predict ST score probabilities
    probabilities = _predict_st_score_probabilities(featured, model_payload)
    
    # Convert probabilities to 0-100 scale.
    # Keep raw calibrated score for audit, then apply a small rank-based uplift
    # so likely winners are more clearly separated in the final score.
    base_scores = np.clip(probabilities * 100.0, 0.0, 100.0)
    out["st_score_pre_model"] = pd.Series(np.round(base_scores, 1), index=out.index)

    rank_scores = (
        pd.Series(probabilities, index=out.index)
        .rank(method="average", pct=True)
        .fillna(0.5)
        .to_numpy(dtype="float64")
        * 100.0
    )
    blended_scores = ((1.0 - ST_RANK_BLEND_WEIGHT) * base_scores) + (ST_RANK_BLEND_WEIGHT * rank_scores)

    # Precision-oriented uplift: reward setups that historically align with
    # faster target hits while penalizing obvious exhaustion/overextension.
    gap_pct = pd.to_numeric(featured.get("feature_gap_pct"), errors="coerce")
    if gap_pct.isna().all():
        gap_pct = pd.to_numeric(featured.get("gap_pct"), errors="coerce")
    exhaustion = pd.to_numeric(featured.get("feature_exhaustion_risk"), errors="coerce")
    extension_penalty = pd.to_numeric(featured.get("score_penalty_extension"), errors="coerce")
    close_vs_prev_high = pd.to_numeric(featured.get("feature_close_vs_prev_high_pct"), errors="coerce")
    p_cont = pd.to_numeric(featured.get("markov_p_continuation"), errors="coerce")
    p_adv = pd.to_numeric(featured.get("markov_p_adverse"), errors="coerce")

    precision_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    precision_bonus += np.where((p_cont.fillna(0.0) >= 0.58) & (p_adv.fillna(0.0) <= 0.22), 2.0, 0.0)
    precision_bonus += np.where(gap_pct.fillna(0.0).between(-0.8, 1.8), 1.5, 0.0)
    precision_bonus += np.where(exhaustion.fillna(0.0) <= 8.5, 2.0, 0.0)
    precision_bonus += np.where(close_vs_prev_high.fillna(0.0) >= -1.2, 1.0, 0.0)
    precision_bonus -= np.where(gap_pct.fillna(0.0) >= 3.5, 2.0, 0.0)
    precision_bonus -= np.where(exhaustion.fillna(0.0) >= 16.0, 2.5, 0.0)
    precision_bonus -= np.where(extension_penalty.fillna(0.0) <= -0.35, 1.5, 0.0)

    final_scores = blended_scores + precision_bonus.to_numpy(dtype="float64")
    out["st_score"] = pd.Series(np.round(np.clip(final_scores, 0.0, 100.0), 1), index=out.index)
    
    # Store encoded markov state as single column (for reference/debugging)
    # We'll store the state name itself, or encode as string of one-hot values
    markov_state_encoded = []
    for idx, row in featured.iterrows():
        state_name = str(row.get("markov_state", "unknown")).strip()
        # Store as comma-separated encoded values for all states
        encoded_vals = [str(int(row.get(f"{level}_encoded", 0))) for level in MARKOV_STATE_LEVELS]
        markov_state_encoded.append("|".join(encoded_vals))
    
    out["markov_state_encoded"] = markov_state_encoded
    
    return out
