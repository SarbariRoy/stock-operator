"""Apply trained ST score model to signals to generate st_score predictions.

The ST (Short-Term) score predicts the probability of hitting a 3% target within
7 days before hitting a 3% stop loss, expressed as a 0-100 score. This module
applies a pre-trained logistic regression model with isotonic calibration.
"""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .markov import build_markov_state_table, load_signal_markov_model
from .stop_risk import _sigmoid, _apply_isotonic_regression

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_ST_SCORE_MODEL_JSON = DATA_DIR / "signal_st_score_model.json"
DEFAULT_ST_SCORE_SVM_MODEL_JSON = DATA_DIR / "signal_st_score_svm_model.json"
DEFAULT_ST_SCORE_RF_MODEL_JSON = DATA_DIR / "signal_st_score_rf_model.json"
DEFAULT_ST_SCORE_XGB_MODEL_JSON = DATA_DIR / "signal_st_score_xgboost_model.json"

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


def _load_json_payload(path: Path) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def load_signal_st_score_model(path: Path = DEFAULT_ST_SCORE_MODEL_JSON) -> dict:
    """Load the default logistic ST score artifact from disk."""
    return _load_json_payload(path)


def load_signal_st_svm_model(path: Path = DEFAULT_ST_SCORE_SVM_MODEL_JSON) -> dict:
    """Load the optional SVM ST score artifact from disk."""
    return _load_json_payload(path)


def load_signal_st_rf_model(path: Path = DEFAULT_ST_SCORE_RF_MODEL_JSON) -> dict:
    """Load the optional Random Forest ST score artifact from disk."""
    return _load_json_payload(path)


def load_signal_st_xgb_model(path: Path = DEFAULT_ST_SCORE_XGB_MODEL_JSON) -> dict:
    """Load the optional XGBoost ST score artifact from disk."""
    return _load_json_payload(path)


def build_st_score_payload(
    *,
    mode: str = "auto",
    logistic_path: Path = DEFAULT_ST_SCORE_MODEL_JSON,
    svm_path: Path = DEFAULT_ST_SCORE_SVM_MODEL_JSON,
    rf_path: Path = DEFAULT_ST_SCORE_RF_MODEL_JSON,
    xgb_path: Path = DEFAULT_ST_SCORE_XGB_MODEL_JSON,
    blend_weight_svm: float = 0.25,
    blend_weight_rf: float = 0.25,
    blend_weight_xgb: float = 0.25,
) -> dict:
    """Build runtime ST payload in logistic/svm/rf/hybrid modes.

    Mode behavior:
    - auto: hybrid4 if all four artifacts exist, else hybrid3, else hybrid, else logistic, else svm, else rf, else xgboost.
    - logistic: force logistic artifact.
    - svm: force svm artifact.
    - rf: force random-forest artifact.
    - xgboost: force xgboost artifact.
    - hybrid: blend both when available, otherwise fallback to available model.
    - hybrid3: blend logistic+svm+rf when available, otherwise fallback to best available mode.
    - hybrid4: blend logistic+svm+rf+xgboost when available, otherwise fallback to best available mode.
    """
    logistic_payload = load_signal_st_score_model(Path(logistic_path))
    svm_payload = load_signal_st_svm_model(Path(svm_path))
    rf_payload = load_signal_st_rf_model(Path(rf_path))
    xgb_payload = load_signal_st_xgb_model(Path(xgb_path))

    has_logistic = isinstance(logistic_payload.get("model"), dict)
    has_svm = isinstance(svm_payload.get("model"), dict)
    has_rf = isinstance(rf_payload.get("model"), dict)
    has_xgb = isinstance(xgb_payload.get("model"), dict)

    requested_mode = str(mode or "auto").strip().lower()
    if requested_mode not in {"auto", "logistic", "svm", "rf", "xgboost", "hybrid", "hybrid3", "hybrid4"}:
        requested_mode = "auto"

    resolved_mode = requested_mode
    if requested_mode == "auto":
        if has_logistic and has_svm and has_rf and has_xgb:
            resolved_mode = "hybrid4"
        elif has_logistic and has_svm and has_rf:
            resolved_mode = "hybrid3"
        elif has_logistic and has_svm:
            resolved_mode = "hybrid"
        elif has_logistic:
            resolved_mode = "logistic"
        elif has_svm:
            resolved_mode = "svm"
        elif has_rf:
            resolved_mode = "rf"
        elif has_xgb:
            resolved_mode = "xgboost"
        else:
            return {}

    if resolved_mode == "hybrid":
        if not (has_logistic and has_svm):
            if has_logistic:
                resolved_mode = "logistic"
            elif has_svm:
                resolved_mode = "svm"
            elif has_rf:
                resolved_mode = "rf"
            elif has_xgb:
                resolved_mode = "xgboost"
            else:
                return {}

    if resolved_mode == "hybrid3":
        if not (has_logistic and has_svm and has_rf):
            if has_logistic and has_svm:
                resolved_mode = "hybrid"
            elif has_logistic:
                resolved_mode = "logistic"
            elif has_svm:
                resolved_mode = "svm"
            elif has_rf:
                resolved_mode = "rf"
            elif has_xgb:
                resolved_mode = "xgboost"
            else:
                return {}

    if resolved_mode == "hybrid4":
        if not (has_logistic and has_svm and has_rf and has_xgb):
            if has_logistic and has_svm and has_rf:
                resolved_mode = "hybrid3"
            elif has_logistic and has_svm:
                resolved_mode = "hybrid"
            elif has_logistic:
                resolved_mode = "logistic"
            elif has_svm:
                resolved_mode = "svm"
            elif has_rf:
                resolved_mode = "rf"
            elif has_xgb:
                resolved_mode = "xgboost"
            else:
                return {}

    if resolved_mode == "logistic":
        if not has_logistic:
            return {}
        payload = dict(logistic_payload)
        payload.setdefault("model_type", "logistic")
        return payload

    if resolved_mode == "svm":
        if not has_svm:
            return {}
        payload = dict(svm_payload)
        payload.setdefault("model_type", "svm")
        return payload

    if resolved_mode == "rf":
        if not has_rf:
            return {}
        payload = dict(rf_payload)
        payload.setdefault("model_type", "rf")
        return payload

    if resolved_mode == "xgboost":
        if not has_xgb:
            return {}
        payload = dict(xgb_payload)
        payload.setdefault("model_type", "xgboost")
        return payload

    if resolved_mode == "hybrid3":
        w_svm = float(blend_weight_svm)
        w_rf = float(blend_weight_rf)
        w_svm = max(0.0, min(1.0, w_svm))
        w_rf = max(0.0, min(1.0, w_rf))
        if (w_svm + w_rf) >= 1.0:
            total = max(1e-9, w_svm + w_rf)
            w_svm = 0.95 * (w_svm / total)
            w_rf = 0.95 * (w_rf / total)
        return {
            "model_type": "hybrid3",
            "blend_weight_svm": w_svm,
            "blend_weight_rf": w_rf,
            "logistic_payload": dict(logistic_payload),
            "svm_payload": dict(svm_payload),
            "rf_payload": dict(rf_payload),
        }

    if resolved_mode == "hybrid4":
        w_svm = max(0.0, min(1.0, float(blend_weight_svm)))
        w_rf = max(0.0, min(1.0, float(blend_weight_rf)))
        w_xgb = max(0.0, min(1.0, float(blend_weight_xgb)))
        if (w_svm + w_rf + w_xgb) >= 1.0:
            total = max(1e-9, (w_svm + w_rf + w_xgb))
            scale = 0.95 / total
            w_svm *= scale
            w_rf *= scale
            w_xgb *= scale
        return {
            "model_type": "hybrid4",
            "blend_weight_svm": w_svm,
            "blend_weight_rf": w_rf,
            "blend_weight_xgb": w_xgb,
            "logistic_payload": dict(logistic_payload),
            "svm_payload": dict(svm_payload),
            "rf_payload": dict(rf_payload),
            "xgb_payload": dict(xgb_payload),
        }

    weight = float(blend_weight_svm)
    weight = max(0.0, min(1.0, weight))
    return {
        "model_type": "hybrid",
        "blend_weight_svm": weight,
        "logistic_payload": dict(logistic_payload),
        "svm_payload": dict(svm_payload),
    }


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


def _predict_st_score_probabilities_svm(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    """Predict ST probabilities using serialized sklearn SVC artifact."""
    if feature_df.empty:
        return np.zeros(0, dtype="float64")

    X = _build_st_feature_frame(feature_df, model_payload)
    serialized = str(model_payload.get("svc_pickled_b64", "") or "").strip()
    if not serialized:
        return np.zeros(len(feature_df), dtype="float64")

    try:
        model_obj = pickle.loads(base64.b64decode(serialized.encode("ascii")))
        raw_probabilities = np.asarray(model_obj.predict_proba(X)[:, 1], dtype="float64")
    except Exception:
        return np.zeros(len(feature_df), dtype="float64")

    upper_bounds = list(model_payload.get("isotonic_upper_bounds", []))
    values = list(model_payload.get("isotonic_values", []))
    return _apply_isotonic_calibration(raw_probabilities, upper_bounds, values)


def _predict_st_score_probabilities_rf(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    """Predict ST probabilities using serialized sklearn RandomForest artifact."""
    if feature_df.empty:
        return np.zeros(0, dtype="float64")

    X = _build_st_feature_frame(feature_df, model_payload)
    serialized = str(model_payload.get("rfc_pickled_b64", "") or "").strip()
    if not serialized:
        return np.zeros(len(feature_df), dtype="float64")

    try:
        model_obj = pickle.loads(base64.b64decode(serialized.encode("ascii")))
        raw_probabilities = np.asarray(model_obj.predict_proba(X)[:, 1], dtype="float64")
    except Exception:
        return np.zeros(len(feature_df), dtype="float64")

    upper_bounds = list(model_payload.get("isotonic_upper_bounds", []))
    values = list(model_payload.get("isotonic_values", []))
    return _apply_isotonic_calibration(raw_probabilities, upper_bounds, values)


def _predict_st_score_probabilities_xgb(feature_df: pd.DataFrame, model_payload: dict) -> np.ndarray:
    """Predict ST probabilities using serialized XGBoost classifier artifact."""
    if feature_df.empty:
        return np.zeros(0, dtype="float64")

    X = _build_st_feature_frame(feature_df, model_payload)
    serialized = str(model_payload.get("xgb_pickled_b64", "") or "").strip()
    if not serialized:
        return np.zeros(len(feature_df), dtype="float64")

    try:
        model_obj = pickle.loads(base64.b64decode(serialized.encode("ascii")))
        raw_probabilities = np.asarray(model_obj.predict_proba(X)[:, 1], dtype="float64")
    except Exception:
        return np.zeros(len(feature_df), dtype="float64")

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
    
    if not isinstance(payload, dict):
        return out

    model_type = str(payload.get("model_type", "logistic") or "logistic").strip().lower()
    if model_type == "hybrid":
        logistic_payload = payload.get("logistic_payload") if isinstance(payload.get("logistic_payload"), dict) else {}
        svm_payload = payload.get("svm_payload") if isinstance(payload.get("svm_payload"), dict) else {}
        logistic_model_payload = logistic_payload.get("model") if isinstance(logistic_payload.get("model"), dict) else {}
        svm_model_payload = svm_payload.get("model") if isinstance(svm_payload.get("model"), dict) else {}
        if not logistic_model_payload and not svm_model_payload:
            return out
    elif model_type == "hybrid3":
        logistic_payload = payload.get("logistic_payload") if isinstance(payload.get("logistic_payload"), dict) else {}
        svm_payload = payload.get("svm_payload") if isinstance(payload.get("svm_payload"), dict) else {}
        rf_payload = payload.get("rf_payload") if isinstance(payload.get("rf_payload"), dict) else {}
        logistic_model_payload = logistic_payload.get("model") if isinstance(logistic_payload.get("model"), dict) else {}
        svm_model_payload = svm_payload.get("model") if isinstance(svm_payload.get("model"), dict) else {}
        rf_model_payload = rf_payload.get("model") if isinstance(rf_payload.get("model"), dict) else {}
        if not logistic_model_payload and not svm_model_payload and not rf_model_payload:
            return out
    elif model_type == "hybrid4":
        logistic_payload = payload.get("logistic_payload") if isinstance(payload.get("logistic_payload"), dict) else {}
        svm_payload = payload.get("svm_payload") if isinstance(payload.get("svm_payload"), dict) else {}
        rf_payload = payload.get("rf_payload") if isinstance(payload.get("rf_payload"), dict) else {}
        xgb_payload = payload.get("xgb_payload") if isinstance(payload.get("xgb_payload"), dict) else {}
        logistic_model_payload = logistic_payload.get("model") if isinstance(logistic_payload.get("model"), dict) else {}
        svm_model_payload = svm_payload.get("model") if isinstance(svm_payload.get("model"), dict) else {}
        rf_model_payload = rf_payload.get("model") if isinstance(rf_payload.get("model"), dict) else {}
        xgb_model_payload = xgb_payload.get("model") if isinstance(xgb_payload.get("model"), dict) else {}
        if not logistic_model_payload and not svm_model_payload and not rf_model_payload and not xgb_model_payload:
            return out
    else:
        model_payload = payload.get("model") if isinstance(payload.get("model"), dict) else {}
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
    if model_type == "svm":
        probabilities = _predict_st_score_probabilities_svm(featured, model_payload)
    elif model_type == "rf":
        probabilities = _predict_st_score_probabilities_rf(featured, model_payload)
    elif model_type == "xgboost":
        probabilities = _predict_st_score_probabilities_xgb(featured, model_payload)
    elif model_type == "hybrid":
        weight_svm = float(payload.get("blend_weight_svm", 0.3) or 0.3)
        weight_svm = max(0.0, min(1.0, weight_svm))
        logistic_probabilities = (
            _predict_st_score_probabilities(featured, logistic_model_payload)
            if logistic_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        svm_probabilities = (
            _predict_st_score_probabilities_svm(featured, svm_model_payload)
            if svm_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        probabilities = ((1.0 - weight_svm) * logistic_probabilities) + (weight_svm * svm_probabilities)
    elif model_type == "hybrid3":
        weight_svm = float(payload.get("blend_weight_svm", 0.3) or 0.3)
        weight_rf = float(payload.get("blend_weight_rf", 0.2) or 0.2)
        weight_svm = max(0.0, min(1.0, weight_svm))
        weight_rf = max(0.0, min(1.0, weight_rf))
        if (weight_svm + weight_rf) >= 1.0:
            total = max(1e-9, weight_svm + weight_rf)
            weight_svm = 0.95 * (weight_svm / total)
            weight_rf = 0.95 * (weight_rf / total)
        weight_log = max(0.0, 1.0 - weight_svm - weight_rf)

        logistic_probabilities = (
            _predict_st_score_probabilities(featured, logistic_model_payload)
            if logistic_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        svm_probabilities = (
            _predict_st_score_probabilities_svm(featured, svm_model_payload)
            if svm_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        rf_probabilities = (
            _predict_st_score_probabilities_rf(featured, rf_model_payload)
            if rf_model_payload
            else np.zeros(len(featured), dtype="float64")
        )

        weights = []
        series = []
        if logistic_model_payload:
            weights.append(weight_log)
            series.append(logistic_probabilities)
        if svm_model_payload:
            weights.append(weight_svm)
            series.append(svm_probabilities)
        if rf_model_payload:
            weights.append(weight_rf)
            series.append(rf_probabilities)

        if not series:
            probabilities = np.zeros(len(featured), dtype="float64")
        else:
            total_w = float(sum(weights))
            if total_w <= 1e-9:
                weights = [1.0 / float(len(series))] * len(series)
            else:
                weights = [float(w) / total_w for w in weights]
            probabilities = np.zeros(len(featured), dtype="float64")
            for w, p in zip(weights, series):
                probabilities += float(w) * p
    elif model_type == "hybrid4":
        weight_svm = max(0.0, min(1.0, float(payload.get("blend_weight_svm", 0.25) or 0.25)))
        weight_rf = max(0.0, min(1.0, float(payload.get("blend_weight_rf", 0.25) or 0.25)))
        weight_xgb = max(0.0, min(1.0, float(payload.get("blend_weight_xgb", 0.25) or 0.25)))
        if (weight_svm + weight_rf + weight_xgb) >= 1.0:
            total = max(1e-9, (weight_svm + weight_rf + weight_xgb))
            scale = 0.95 / total
            weight_svm *= scale
            weight_rf *= scale
            weight_xgb *= scale
        weight_log = max(0.0, 1.0 - weight_svm - weight_rf - weight_xgb)

        logistic_probabilities = (
            _predict_st_score_probabilities(featured, logistic_model_payload)
            if logistic_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        svm_probabilities = (
            _predict_st_score_probabilities_svm(featured, svm_model_payload)
            if svm_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        rf_probabilities = (
            _predict_st_score_probabilities_rf(featured, rf_model_payload)
            if rf_model_payload
            else np.zeros(len(featured), dtype="float64")
        )
        xgb_probabilities = (
            _predict_st_score_probabilities_xgb(featured, xgb_model_payload)
            if xgb_model_payload
            else np.zeros(len(featured), dtype="float64")
        )

        weights = []
        series = []
        if logistic_model_payload:
            weights.append(weight_log)
            series.append(logistic_probabilities)
        if svm_model_payload:
            weights.append(weight_svm)
            series.append(svm_probabilities)
        if rf_model_payload:
            weights.append(weight_rf)
            series.append(rf_probabilities)
        if xgb_model_payload:
            weights.append(weight_xgb)
            series.append(xgb_probabilities)

        if not series:
            probabilities = np.zeros(len(featured), dtype="float64")
        else:
            total_w = float(sum(weights))
            if total_w <= 1e-9:
                weights = [1.0 / float(len(series))] * len(series)
            else:
                weights = [float(w) / total_w for w in weights]
            probabilities = np.zeros(len(featured), dtype="float64")
            for w, p in zip(weights, series):
                probabilities += float(w) * p
    else:
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

    # Extract key features for multi-faceted scoring boosters
    gap_pct = pd.to_numeric(featured.get("feature_gap_pct"), errors="coerce")
    if isinstance(gap_pct, (float, int, np.floating)) or (isinstance(gap_pct, pd.Series) and gap_pct.isna().all()):
        gap_pct = pd.to_numeric(featured.get("gap_pct"), errors="coerce")
    if not isinstance(gap_pct, pd.Series):
        gap_pct = pd.Series(gap_pct, index=featured.index)
        
    exhaustion = pd.to_numeric(featured.get("feature_exhaustion_risk"), errors="coerce")
    if not isinstance(exhaustion, pd.Series):
        exhaustion = pd.Series(exhaustion, index=featured.index)
        
    extension_penalty = pd.to_numeric(featured.get("score_penalty_extension"), errors="coerce")
    if not isinstance(extension_penalty, pd.Series):
        extension_penalty = pd.Series(extension_penalty, index=featured.index)
        
    close_vs_prev_high = pd.to_numeric(featured.get("feature_close_vs_prev_high_pct"), errors="coerce")
    if not isinstance(close_vs_prev_high, pd.Series):
        close_vs_prev_high = pd.Series(close_vs_prev_high, index=featured.index)
        
    p_cont = pd.to_numeric(featured.get("markov_p_continuation"), errors="coerce")
    if not isinstance(p_cont, pd.Series):
        p_cont = pd.Series(p_cont, index=featured.index)
        
    p_adv = pd.to_numeric(featured.get("markov_p_adverse"), errors="coerce")
    if not isinstance(p_adv, pd.Series):
        p_adv = pd.Series(p_adv, index=featured.index)
        
    consensus_count = pd.to_numeric(featured.get("consensus_count"), errors="coerce")
    if not isinstance(consensus_count, pd.Series):
        consensus_count = pd.Series(consensus_count, index=featured.index)
    consensus_count = consensus_count.fillna(1.0)
    
    markov_state = featured.get("markov_state", pd.Series("", index=featured.index))
    if not isinstance(markov_state, pd.Series):
        markov_state = pd.Series(markov_state, index=featured.index)
    markov_state = markov_state.astype(str).str.strip()
    
    range_vs_atr = pd.to_numeric(featured.get("feature_range_vs_atr"), errors="coerce")
    if not isinstance(range_vs_atr, pd.Series):
        range_vs_atr = pd.Series(range_vs_atr, index=featured.index)
    range_vs_atr = range_vs_atr.fillna(1.0)
    
    regime_median_ret = pd.to_numeric(featured.get("regime_median_ret_20d_pct"), errors="coerce")
    if not isinstance(regime_median_ret, pd.Series):
        regime_median_ret = pd.Series(regime_median_ret, index=featured.index)
    regime_median_ret = regime_median_ret.fillna(0.0)
    
    regime_pct_above_sma50 = pd.to_numeric(featured.get("regime_pct_above_sma50"), errors="coerce")
    if not isinstance(regime_pct_above_sma50, pd.Series):
        regime_pct_above_sma50 = pd.Series(regime_pct_above_sma50, index=featured.index)
    regime_pct_above_sma50 = regime_pct_above_sma50.fillna(50.0)

    # ========== BOOSTER 1: Enhanced Markov Regime Confidence Booster ==========
    # Reward high conviction trend continuations and fresh breakouts
    markov_confidence_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    
    # Very high continuation probability + low adverse risk = strong trend conviction
    markov_confidence_bonus += np.where(
        (p_cont.fillna(0.0) >= 0.65) & (p_adv.fillna(0.0) <= 0.20),
        8.0,  # Highest confidence
        0.0
    )
    
    # Fresh breakout with good continuation = early in move, +7 pts
    markov_confidence_bonus += np.where(
        (markov_state == "fresh_breakout") & (p_cont.fillna(0.0) >= 0.60),
        np.where(p_cont.fillna(0.0) >= 0.68, 7.0, 5.0),
        0.0
    )
    
    # Constructive trend with moderate continuation = +3 pts
    markov_confidence_bonus += np.where(
        (markov_state == "constructive_trend") & (p_cont.fillna(0.0) >= 0.58),
        3.0,
        0.0
    )
    
    # Penalize extended breakout or breakdown risk
    markov_confidence_bonus -= np.where(
        markov_state.isin(["extended_breakout", "breakdown_risk"]),
        2.0,
        0.0
    )

    # ========== BOOSTER 2: Pattern Consensus Booster ==========
    # Multiple patterns agreeing = higher conviction
    consensus_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    consensus_bonus += np.where(consensus_count >= 3, 6.0, 0.0)  # 3+ patterns
    consensus_bonus += np.where(consensus_count == 2, 3.0, 0.0)   # 2 patterns
    # Single pattern gets 0 bonus

    # ========== BOOSTER 3: Entry Quality Score Booster ==========
    # Reward clean, non-crowded entries
    entry_quality_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    
    # Clean entry: not gapped up, not gapped down excessively
    entry_quality_bonus += np.where(
        gap_pct.fillna(0.0).between(-0.5, 1.5),
        3.0,
        0.0
    )
    
    # Not over-extended internally
    entry_quality_bonus += np.where(
        exhaustion.fillna(0.0) < 8.0,
        2.0,
        0.0
    )
    
    # Entry near recent highs (not far below)
    entry_quality_bonus += np.where(
        close_vs_prev_high.fillna(0.0) >= -1.0,
        1.0,
        0.0
    )
    
    # Penalize gap-shocked entries
    entry_quality_bonus -= np.where(
        gap_pct.fillna(0.0) >= 3.0,
        2.0,
        0.0
    )

    # ========== BOOSTER 4: Volatility Regime Adapter ==========
    # Trending market with controlled intraday volatility
    volatility_regime_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    
    # Trending market (positive 20d median return) with controlled intraday moves
    volatility_regime_bonus += np.where(
        (regime_median_ret.fillna(0.0) > 1.5) & (range_vs_atr.fillna(1.0) < 1.2),
        4.0,
        0.0
    )
    
    # Healthy uptrend environment (most stocks above SMA50)
    volatility_regime_bonus += np.where(
        regime_pct_above_sma50.fillna(50.0) > 75.0,
        2.0,
        0.0
    )
    
    # Penalize negative regime
    volatility_regime_bonus -= np.where(
        regime_median_ret.fillna(0.0) < -1.5,
        2.0,
        0.0
    )

    # ========== LEGACY PRECISION BONUS (kept for backward compatibility) ==========
    precision_bonus = pd.Series(0.0, index=out.index, dtype="float64")
    precision_bonus += np.where((p_cont.fillna(0.0) >= 0.58) & (p_adv.fillna(0.0) <= 0.22), 2.0, 0.0)
    precision_bonus += np.where(gap_pct.fillna(0.0).between(-0.8, 1.8), 1.5, 0.0)
    precision_bonus += np.where(exhaustion.fillna(0.0) <= 8.5, 2.0, 0.0)
    precision_bonus += np.where(close_vs_prev_high.fillna(0.0) >= -1.2, 1.0, 0.0)
    precision_bonus -= np.where(gap_pct.fillna(0.0) >= 3.5, 2.0, 0.0)
    precision_bonus -= np.where(exhaustion.fillna(0.0) >= 16.0, 2.5, 0.0)
    precision_bonus -= np.where(extension_penalty.fillna(0.0) <= -0.35, 1.5, 0.0)

    # ========== COMBINE ALL BOOSTERS ==========
    total_boosters = (
        markov_confidence_bonus
        + consensus_bonus
        + entry_quality_bonus
        + volatility_regime_bonus
        + precision_bonus
    )
    
    # Cap total booster contribution to avoid extreme outliers
    # Natural range: -4 to +24, clip to [-6, 18] for safety margin
    total_boosters = np.clip(total_boosters.to_numpy(dtype="float64"), -6.0, 18.0)

    final_scores = blended_scores + total_boosters
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
