"""Train an SVM model for ST scoring and save artifact as JSON.

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
from typing import Any

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
DEFAULT_SIGNAL_ST_SCORE_SVM_MODEL_JSON = DATA_DIR / "st_signal_st_score_svm_model.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train calibrated SVM model for ST score objective"
    )
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument(
        "--training-data",
        type=str,
        default="",
        help="Optional shared training artifact with precomputed features and ST labels",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNAL_ST_SCORE_SVM_MODEL_JSON))
    parser.add_argument("--target-pct", type=float, default=ST_TARGET_PCT)
    parser.add_argument("--stop-pct", type=float, default=ST_STOP_PCT)
    parser.add_argument("--hold-days", type=int, default=ST_HOLD_DAYS)
    parser.add_argument("--train-start-date", type=str, default="")
    parser.add_argument("--train-end-date", type=str, default="")
    parser.add_argument("--allow-partial-horizon", action="store_true")
    parser.add_argument("--recency-half-life-months", type=float, default=0.0)
    parser.add_argument("--svm-kernel", type=str, default="rbf", choices=("linear", "rbf", "poly", "sigmoid"))
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--svm-degree", type=int, default=3)
    return parser.parse_args()


def _prepare_numeric_and_family_matrix(
    frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    family_levels: list[str],
) -> tuple[np.ndarray, dict, dict, dict]:
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
        "n_support_vectors": int(payload.get("n_support_vectors", 0)),
    }


def _build_st_svm_model(
    feature_df: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    family_levels: list[str] | None = None,
    svm_kernel: str = "rbf",
    svm_c: float = 1.0,
    svm_gamma: str = "scale",
    svm_degree: int = 3,
) -> dict:
    numeric_features = [
        feature for feature in (numeric_features or ST_NUMERIC_FEATURES) if feature in feature_df.columns
    ]
    family_levels = list(family_levels or ST_FAMILY_LEVELS)

    working = feature_df.copy()
    working = working.dropna(subset=["ticker", "signal_date", "pattern_family", ST_TARGET])
    if working.empty:
        return {
            "positive_rate": 0.0,
            "model_algorithm": "svm",
            "numeric_features": numeric_features,
            "family_levels": family_levels,
            "impute_medians": {},
            "scaler_means": {},
            "scaler_stds": {},
            "svc_pickled_b64": "",
            "isotonic_upper_bounds": [],
            "isotonic_values": [],
            "n_support_vectors": 0,
            "svm_kernel": svm_kernel,
            "svm_c": float(svm_c),
            "svm_gamma": svm_gamma,
            "svm_degree": int(svm_degree),
        }

    weight_source = working["sample_weight"] if "sample_weight" in working.columns else pd.Series(1.0, index=working.index)
    sample_weight = pd.to_numeric(weight_source, errors="coerce").fillna(0.0).astype("float64").to_numpy()

    X, impute_medians, scaler_means, scaler_stds = _prepare_numeric_and_family_matrix(
        working,
        numeric_features=numeric_features,
        family_levels=family_levels,
    )
    y = pd.to_numeric(working[ST_TARGET], errors="coerce").fillna(0.0).astype("int64").to_numpy()

    gamma_param: str | float
    try:
        gamma_param = float(svm_gamma)
    except (TypeError, ValueError):
        gamma_param = str(svm_gamma)

    try:
        from sklearn.svm import SVC  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for SVM training. Install it with: "
            "python -m pip install scikit-learn"
        ) from exc

    svc: Any = SVC(
        kernel=str(svm_kernel),
        C=float(svm_c),
        gamma=gamma_param,
        degree=int(svm_degree),
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    if float(sample_weight.sum()) > 0:
        svc.fit(X, y, sample_weight=sample_weight)
    else:
        svc.fit(X, y)

    raw_probabilities = np.asarray(svc.predict_proba(X)[:, 1], dtype="float64")
    isotonic_upper_bounds, isotonic_values = _fit_isotonic_regression(
        raw_probabilities,
        y.astype("float64"),
        sample_weight=sample_weight,
    )

    model_bytes = pickle.dumps(svc, protocol=pickle.HIGHEST_PROTOCOL)
    model_b64 = base64.b64encode(model_bytes).decode("ascii")

    return {
        "positive_rate": round(
            float(np.average(y, weights=sample_weight)) if len(y) and float(sample_weight.sum()) > 0 else float(np.mean(y)),
            10,
        ),
        "model_algorithm": "svm",
        "numeric_features": numeric_features,
        "family_levels": family_levels,
        "impute_medians": impute_medians,
        "scaler_means": scaler_means,
        "scaler_stds": scaler_stds,
        "svc_pickled_b64": model_b64,
        "isotonic_upper_bounds": isotonic_upper_bounds,
        "isotonic_values": isotonic_values,
        "n_support_vectors": int(np.sum(getattr(svc, "n_support_", np.array([0])))),
        "svm_kernel": str(svm_kernel),
        "svm_c": round(float(svm_c), 8),
        "svm_gamma": svm_gamma,
        "svm_degree": int(svm_degree),
    }


def compute_st_svm_model_from_training_data(
    training_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    hold_days: int,
    require_full_horizon: bool,
    svm_kernel: str,
    svm_c: float,
    svm_gamma: str,
    svm_degree: int,
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
            "model_type": "svm",
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "hold_days": int(hold_days),
            "numeric_features": [],
            "include_family_features": True,
            "require_full_horizon": bool(require_full_horizon),
            "model": {},
        }

    selected_numeric_features = [feature for feature in ST_NUMERIC_FEATURES if feature in merged.columns]
    model_payload = _build_st_svm_model(
        merged,
        numeric_features=selected_numeric_features,
        family_levels=ST_FAMILY_LEVELS,
        svm_kernel=svm_kernel,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        svm_degree=svm_degree,
    )

    return {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "model_type": "svm",
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

        payload = compute_st_svm_model_from_training_data(
            training,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            require_full_horizon=require_full_horizon,
            svm_kernel=str(args.svm_kernel),
            svm_c=float(args.svm_c),
            svm_gamma=str(args.svm_gamma),
            svm_degree=int(args.svm_degree),
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
        featured = compute_st_features(signals, prices)
        featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")
        merged = featured.merge(labels, on=["ticker", "signal_date", "pattern_family"], how="inner")

        payload = compute_st_svm_model_from_training_data(
            merged,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            require_full_horizon=require_full_horizon,
            svm_kernel=str(args.svm_kernel),
            svm_c=float(args.svm_c),
            svm_gamma=str(args.svm_gamma),
            svm_degree=int(args.svm_degree),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nST SVM Model Summary:")
    print(f"  Computed at: {payload.get('computed_at', 'N/A')}")
    print(f"  Signals analyzed: {payload.get('signals_analyzed', 0)}")
    print(f"  Target %: {payload.get('target_pct', 0)}%")
    print(f"  Stop %: {payload.get('stop_pct', 0)}%")
    print(f"  Hold days: {payload.get('hold_days', 0)}")
    summary = payload.get("model_summary", {})
    print(f"  Target hit rate: {summary.get('positive_rate', 0.0):.4f}")
    print(f"  Calibration points: {summary.get('calibration_points', 0)}")
    print(f"  Support vectors: {summary.get('n_support_vectors', 0)}")
    print(f"SVM model saved to: {out_path}")


if __name__ == "__main__":
    main()
