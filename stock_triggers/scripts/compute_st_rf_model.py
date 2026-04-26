"""Train a Random Forest model for ST scoring and save artifact as JSON.

The model predicts the probability of hitting a target before stop loss within
an ST horizon, then applies isotonic calibration for smoother probabilities.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_st_score_model import (  # noqa: E402
    DEFAULT_PRICES,
    DEFAULT_SIGNALS,
    DEFAULT_TRAINING_DATA,
    ST_FAMILY_LEVELS,
    ST_NUMERIC_FEATURES,
    ST_TARGET,
    ST_TARGET_PCT,
    ST_STOP_PCT,
    ST_HOLD_DAYS,
    _load_training_data,
    compute_st_features,
    compute_st_outcome_labels,
)
from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, parse_optional_date  # noqa: E402
from stock_triggers.ui.patterns.stop_risk import _fit_isotonic_regression  # noqa: E402

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_ST_SCORE_RF_MODEL_JSON = DATA_DIR / "signal_st_score_rf_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated Random Forest model for ST score objective")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument(
        "--training-data",
        type=str,
        default="",
        help="Optional shared training artifact with precomputed features and ST labels",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNAL_ST_SCORE_RF_MODEL_JSON))
    parser.add_argument("--target-pct", type=float, default=ST_TARGET_PCT)
    parser.add_argument("--stop-pct", type=float, default=ST_STOP_PCT)
    parser.add_argument("--hold-days", type=int, default=ST_HOLD_DAYS)
    parser.add_argument("--train-start-date", type=str, default="")
    parser.add_argument("--train-end-date", type=str, default="")
    parser.add_argument("--allow-partial-horizon", action="store_true")
    parser.add_argument("--recency-half-life-months", type=float, default=0.0)
    parser.add_argument("--rf-n-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=8)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=8)
    parser.add_argument("--rf-min-samples-split", type=int, default=20)
    return parser.parse_args()


def _prepare_numeric_and_family_matrix(
    frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    family_levels: list[str],
) -> tuple[np.ndarray, dict[str, float], dict[str, float], dict[str, float]]:
    impute_medians: dict[str, float] = {}
    scaler_means: dict[str, float] = {}
    scaler_stds: dict[str, float] = {}

    parts: list[np.ndarray] = []
    for feature_name in numeric_features:
        series = pd.to_numeric(frame[feature_name], errors="coerce")
        median_value = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median_value).astype("float64")
        mean_value = float(filled.mean()) if len(filled) else 0.0
        std_value = float(filled.std(ddof=0)) if len(filled) else 1.0
        if std_value <= 1e-8:
            std_value = 1.0

        impute_medians[feature_name] = round(median_value, 10)
        scaler_means[feature_name] = round(mean_value, 10)
        scaler_stds[feature_name] = round(std_value, 10)

        normalized = ((filled - mean_value) / std_value).to_numpy().reshape(-1, 1)
        parts.append(normalized)

    family_series = frame.get("pattern_family", pd.Series("", index=frame.index)).astype(str).str.strip().str.upper()
    for family in family_levels:
        parts.append((family_series == family).astype("float64").to_numpy().reshape(-1, 1))

    X = np.hstack(parts).astype("float64") if parts else np.zeros((len(frame), 0), dtype="float64")
    return X, impute_medians, scaler_means, scaler_stds


def _summarize_model(payload: dict) -> dict:
    return {
        "positive_rate": round(float(payload.get("positive_rate", 0.0)), 4),
        "calibration_points": int(len(payload.get("isotonic_upper_bounds", []))),
    }


def _build_st_rf_model(
    feature_df: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    family_levels: list[str] | None = None,
    rf_n_estimators: int = 400,
    rf_max_depth: int = 8,
    rf_min_samples_leaf: int = 8,
    rf_min_samples_split: int = 20,
) -> dict:
    numeric_features = [feature for feature in (numeric_features or ST_NUMERIC_FEATURES) if feature in feature_df.columns]
    family_levels = list(family_levels or ST_FAMILY_LEVELS)

    working = feature_df.copy()
    working = working.dropna(subset=["ticker", "signal_date", "pattern_family", ST_TARGET])
    if working.empty:
        return {
            "positive_rate": 0.0,
            "model_algorithm": "random_forest",
            "numeric_features": numeric_features,
            "family_levels": family_levels,
            "impute_medians": {},
            "scaler_means": {},
            "scaler_stds": {},
            "rfc_pickled_b64": "",
            "isotonic_upper_bounds": [],
            "isotonic_values": [],
            "rf_n_estimators": int(rf_n_estimators),
            "rf_max_depth": int(rf_max_depth),
            "rf_min_samples_leaf": int(rf_min_samples_leaf),
            "rf_min_samples_split": int(rf_min_samples_split),
        }

    weight_source = working["sample_weight"] if "sample_weight" in working.columns else pd.Series(1.0, index=working.index)
    sample_weight = pd.to_numeric(weight_source, errors="coerce").fillna(0.0).astype("float64").to_numpy()

    X, impute_medians, scaler_means, scaler_stds = _prepare_numeric_and_family_matrix(
        working,
        numeric_features=numeric_features,
        family_levels=family_levels,
    )
    y = pd.to_numeric(working[ST_TARGET], errors="coerce").fillna(0.0).astype("int64").to_numpy()

    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for RF training. Install it with: "
            "python -m pip install scikit-learn"
        ) from exc

    clf = RandomForestClassifier(
        n_estimators=int(rf_n_estimators),
        max_depth=int(rf_max_depth),
        min_samples_leaf=int(rf_min_samples_leaf),
        min_samples_split=int(rf_min_samples_split),
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=4,
    )

    if float(sample_weight.sum()) > 0:
        clf.fit(X, y, sample_weight=sample_weight)
    else:
        clf.fit(X, y)

    raw_probabilities = np.asarray(clf.predict_proba(X)[:, 1], dtype="float64")
    isotonic_upper_bounds, isotonic_values = _fit_isotonic_regression(
        raw_probabilities,
        y.astype("float64"),
        sample_weight=sample_weight,
    )

    model_bytes = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    model_b64 = base64.b64encode(model_bytes).decode("ascii")

    return {
        "positive_rate": round(
            float(np.average(y, weights=sample_weight)) if len(y) and float(sample_weight.sum()) > 0 else float(np.mean(y)),
            10,
        ),
        "model_algorithm": "random_forest",
        "numeric_features": numeric_features,
        "family_levels": family_levels,
        "impute_medians": impute_medians,
        "scaler_means": scaler_means,
        "scaler_stds": scaler_stds,
        "rfc_pickled_b64": model_b64,
        "isotonic_upper_bounds": isotonic_upper_bounds,
        "isotonic_values": isotonic_values,
        "rf_n_estimators": int(rf_n_estimators),
        "rf_max_depth": int(rf_max_depth),
        "rf_min_samples_leaf": int(rf_min_samples_leaf),
        "rf_min_samples_split": int(rf_min_samples_split),
    }


def compute_st_rf_model_from_training_data(
    training_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    hold_days: int,
    require_full_horizon: bool,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_min_samples_leaf: int,
    rf_min_samples_split: int,
) -> dict:
    required_columns = {"ticker", "signal_date", "pattern_family", ST_TARGET}
    missing = sorted(required_columns - set(training_df.columns))
    if missing:
        raise SystemExit(f"Training data missing required ST columns: {missing}")

    merged = training_df.copy()
    merged["signal_date"] = pd.to_datetime(merged["signal_date"], errors="coerce").dt.date.astype("string")

    if bool(require_full_horizon) and "bars_available_forward" in merged.columns:
        bars_available = pd.to_numeric(merged["bars_available_forward"], errors="coerce")
        merged = merged.loc[bars_available >= int(hold_days)].copy()

    merged.dropna(subset=["ticker", "signal_date", "pattern_family", ST_TARGET], inplace=True)
    if merged.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "model_type": "rf",
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "hold_days": int(hold_days),
            "numeric_features": [],
            "include_family_features": True,
            "require_full_horizon": bool(require_full_horizon),
            "model": {},
        }

    selected_numeric_features = [feature for feature in ST_NUMERIC_FEATURES if feature in merged.columns]
    model_payload = _build_st_rf_model(
        merged,
        numeric_features=selected_numeric_features,
        family_levels=ST_FAMILY_LEVELS,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_min_samples_split=rf_min_samples_split,
    )

    return {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "model_type": "rf",
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "hold_days": int(hold_days),
        "numeric_features": selected_numeric_features,
        "include_family_features": True,
        "require_full_horizon": bool(require_full_horizon),
        "model": model_payload,
        "model_summary": _summarize_model(model_payload),
    }


def main() -> None:
    args = parse_args()
    prices_path = Path(args.prices)
    signals_path = Path(args.signals)
    training_data_path = Path(args.training_data) if args.training_data else DEFAULT_TRAINING_DATA
    out_path = Path(args.out)

    require_full_horizon = not bool(args.allow_partial_horizon)

    if training_data_path.exists():
        print(f"Loading training artifact from {training_data_path} ...")
        training = _load_training_data(training_data_path)
        print(f"  {len(training):,} rows")

        train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
        train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
        training = filter_by_date_window(training, date_col="signal_date", start_date=train_start_date, end_date=train_end_date)

        if ST_TARGET not in training.columns:
            print("Computing ST labels from prices...")
            if not prices_path.exists():
                raise SystemExit(f"Prices file not found: {prices_path}")
            prices = pd.read_csv(prices_path, parse_dates=["Date"])
            st_labels = compute_st_outcome_labels(
                training,
                prices,
                target_pct=float(args.target_pct),
                stop_pct=float(args.stop_pct),
                hold_days=int(args.hold_days),
                require_full_horizon=require_full_horizon,
            )
            training = training.merge(st_labels, on=["ticker", "signal_date", "pattern_family"], how="inner")
            print(f"  After merging labels: {len(training):,} rows")

        if float(args.recency_half_life_months) > 0:
            training = add_recency_weights(training, date_col="signal_date", half_life_months=float(args.recency_half_life_months))

        payload = compute_st_rf_model_from_training_data(
            training,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            require_full_horizon=require_full_horizon,
            rf_n_estimators=int(args.rf_n_estimators),
            rf_max_depth=int(args.rf_max_depth),
            rf_min_samples_leaf=int(args.rf_min_samples_leaf),
            rf_min_samples_split=int(args.rf_min_samples_split),
        )
    else:
        if not prices_path.exists():
            raise SystemExit(f"Prices file not found: {prices_path}")
        if not signals_path.exists():
            raise SystemExit(f"Signals file not found: {signals_path}")

        print(f"Loading prices from {prices_path} ...")
        prices = pd.read_csv(prices_path, parse_dates=["Date"])

        print(f"Loading signals from {signals_path} ...")
        signals = pd.read_csv(signals_path)
        if "signal_date" in signals.columns:
            signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")

        train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
        train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
        signals = filter_by_date_window(signals, date_col="signal_date", start_date=train_start_date, end_date=train_end_date)

        if float(args.recency_half_life_months) > 0:
            signals = add_recency_weights(signals, date_col="signal_date", half_life_months=float(args.recency_half_life_months))

        labels = compute_st_outcome_labels(
            signals,
            prices,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            require_full_horizon=require_full_horizon,
        )
        features = compute_st_features(signals, prices)
        features["signal_date"] = pd.to_datetime(features["signal_date"], errors="coerce").dt.date.astype("string")
        merged = features.merge(labels, on=["ticker", "signal_date", "pattern_family"], how="inner")

        selected_numeric_features = [feature for feature in ST_NUMERIC_FEATURES if feature in merged.columns]
        model_payload = _build_st_rf_model(
            merged,
            numeric_features=selected_numeric_features,
            family_levels=ST_FAMILY_LEVELS,
            rf_n_estimators=int(args.rf_n_estimators),
            rf_max_depth=int(args.rf_max_depth),
            rf_min_samples_leaf=int(args.rf_min_samples_leaf),
            rf_min_samples_split=int(args.rf_min_samples_split),
        )

        payload = {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": int(len(merged)),
            "model_type": "rf",
            "target_pct": float(args.target_pct),
            "stop_pct": float(args.stop_pct),
            "hold_days": int(args.hold_days),
            "numeric_features": selected_numeric_features,
            "include_family_features": True,
            "require_full_horizon": bool(require_full_horizon),
            "model": model_payload,
            "model_summary": _summarize_model(model_payload),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    summary = payload.get("model_summary", {})
    print("\nST RF Model Summary:")
    print(f"  Computed at: {payload.get('computed_at', 'N/A')}")
    print(f"  Signals analyzed: {payload.get('signals_analyzed', 0)}")
    print(f"  Target %: {payload.get('target_pct', 0)}%")
    print(f"  Stop %: {payload.get('stop_pct', 0)}%")
    print(f"  Hold days: {payload.get('hold_days', 0)}")
    print(f"  Target hit rate: {summary.get('positive_rate', 0.0):.4f}")
    print(f"  Calibration points: {summary.get('calibration_points', 0)}")
    print(f"  Numeric features: {len(payload.get('numeric_features', []))}")
    print(f"\nSaved ST RF model to: {out_path}")


if __name__ == "__main__":
    main()
