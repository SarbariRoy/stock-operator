"""Train a separate monotonic stop-risk model from historical signals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.ui.patterns.stop_risk import (
    DEFAULT_SIGNAL_STOP_RISK_MODEL_JSON,
    STOP_RISK_FAMILY_LEVELS,
    STOP_RISK_FEATURE_SET_PRESETS,
    STOP_RISK_NUMERIC_FEATURES,
    STOP_RISK_TARGETS,
    _fit_isotonic_regression,
    _fit_logistic_regression,
    _sigmoid,
    prepare_stop_risk_features,
)

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a calibrated stop-risk model from historical signal rows")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNAL_STOP_RISK_MODEL_JSON))
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--max-hold-days", type=int, default=30)
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument("--recent-signal-lookback-days", type=int, default=5)
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=sorted(STOP_RISK_FEATURE_SET_PRESETS),
        default="full",
        help="Named numeric feature subset to use for training",
    )
    parser.add_argument(
        "--disable-family-features",
        action="store_true",
        help="Do not include pattern-family one-hot features in the trained model",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="",
        help="Only use signals on or before this date for training (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--allow-partial-horizon",
        action="store_true",
        help="Allow rows with fewer than max-hold-days of future prices when building labels",
    )
    return parser.parse_args()


def _resolve_price_history(grouped: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in grouped:
        return grouped[clean]
    if clean.endswith(".NS"):
        return grouped.get(clean[:-3])
    return grouped.get(clean + ".NS")


def compute_stop_event_labels(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    require_full_horizon: bool = True,
) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices.groupby("Ticker", sort=False)}

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig.get("ticker", "")).strip()
        family = str(sig.get("pattern_family", "")).strip().upper()
        signal_date = pd.to_datetime(sig.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(sig.get("entry_price"), errors="coerce")
        if not ticker or not family or pd.isna(signal_date) or pd.isna(entry_price) or float(entry_price) <= 0:
            continue

        hist = _resolve_price_history(grouped, ticker)
        if hist is None:
            continue

        future = hist[hist["Date"] > signal_date].head(int(max_hold_days)).copy()
        if future.empty:
            continue
        if bool(require_full_horizon) and len(future) < int(max_hold_days):
            continue

        stop_price = pd.to_numeric(sig.get("stop_price"), errors="coerce")
        if pd.isna(stop_price) or float(stop_price) <= 0 or float(stop_price) >= float(entry_price):
            stop_price = float(entry_price) * (1.0 - float(stop_pct) / 100.0)
        else:
            stop_price = float(stop_price)
        target_price = float(entry_price) * (1.0 + float(target_pct) / 100.0)

        stop_before_target = 0
        stop_within_5d = 0
        gap_through_stop = 0
        mae_exceeds_stop = 0

        if not future.empty:
            first_bar = future.iloc[0]
            try:
                gap_through_stop = int(float(first_bar.get("Open", np.nan)) <= stop_price)
            except (TypeError, ValueError):
                gap_through_stop = 0

        lowest_low = pd.to_numeric(future.get("Low"), errors="coerce").min()
        mae_exceeds_stop = int(pd.notna(lowest_low) and float(lowest_low) <= stop_price)

        future_5d = future.head(5)
        for _, bar in future_5d.iterrows():
            low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
            high_value = pd.to_numeric(bar.get("High"), errors="coerce")
            if pd.notna(low_value) and float(low_value) <= stop_price:
                stop_within_5d = 1
                break
            if pd.notna(high_value) and float(high_value) >= target_price:
                break

        for _, bar in future.iterrows():
            low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
            high_value = pd.to_numeric(bar.get("High"), errors="coerce")
            if pd.notna(low_value) and float(low_value) <= stop_price:
                stop_before_target = 1
                break
            if pd.notna(high_value) and float(high_value) >= target_price:
                break

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date.date().isoformat(),
                "pattern_family": family,
                "stop_before_target": int(stop_before_target),
                "stop_within_5d": int(stop_within_5d),
                "gap_through_stop": int(gap_through_stop),
                "mae_exceeds_stop": int(mae_exceeds_stop),
            }
        )

    return pd.DataFrame(rows)


def _summarize_target(payload: dict) -> dict:
    return {
        "positive_rate": round(float(payload.get("positive_rate", 0.0)), 4),
        "calibration_points": int(len(payload.get("isotonic_upper_bounds", []))),
    }


def _build_target_model(
    feature_df: pd.DataFrame,
    target_name: str,
    *,
    numeric_features: list[str] | None = None,
    family_levels: list[str] | None = None,
) -> dict:
    numeric_features = [
        feature for feature in (numeric_features or STOP_RISK_NUMERIC_FEATURES) if feature in feature_df.columns
    ]
    family_levels = list(family_levels or STOP_RISK_FAMILY_LEVELS)
    working = feature_df.copy()

    impute_medians: dict[str, float] = {}
    scaler_means: dict[str, float] = {}
    scaler_stds: dict[str, float] = {}
    numeric_parts: list[np.ndarray] = []

    for feature_name in numeric_features:
        series = pd.to_numeric(working[feature_name], errors="coerce")
        median_value = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median_value).astype("float64")
        mean_value = float(filled.mean()) if len(filled) else 0.0
        std_value = float(filled.std(ddof=0)) if len(filled) else 1.0
        if std_value <= 1e-8:
            std_value = 1.0
        impute_medians[feature_name] = round(median_value, 10)
        scaler_means[feature_name] = round(mean_value, 10)
        scaler_stds[feature_name] = round(std_value, 10)
        numeric_parts.append((((filled - mean_value) / std_value).to_numpy()).reshape(-1, 1))

    family_series = working.get("pattern_family", pd.Series("", index=working.index)).astype(str).str.strip().str.upper()
    family_parts = [
        (family_series == family).astype("float64").to_numpy().reshape(-1, 1)
        for family in family_levels
    ]

    X_parts = numeric_parts + family_parts
    X = np.hstack(X_parts).astype("float64") if X_parts else np.zeros((len(working), 0), dtype="float64")
    y = pd.to_numeric(working[target_name], errors="coerce").fillna(0.0).astype("float64").to_numpy()

    coefficients, intercept = _fit_logistic_regression(X, y)
    raw_probabilities = _sigmoid((X @ coefficients) + intercept)
    isotonic_upper_bounds, isotonic_values = _fit_isotonic_regression(raw_probabilities, y)

    return {
        "positive_rate": round(float(y.mean()) if len(y) else 0.0, 10),
        "numeric_features": numeric_features,
        "family_levels": family_levels,
        "impute_medians": impute_medians,
        "scaler_means": scaler_means,
        "scaler_stds": scaler_stds,
        "coefficients": [round(float(value), 10) for value in coefficients.tolist()],
        "intercept": round(float(intercept), 10),
        "isotonic_upper_bounds": isotonic_upper_bounds,
        "isotonic_values": isotonic_values,
    }


def compute_stop_risk_model(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    recent_signal_lookback_days: int,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    numeric_features: list[str] | None = None,
    include_family_features: bool = True,
    feature_set_name: str | None = None,
    require_full_horizon: bool = True,
) -> dict:
    featured = prepare_stop_risk_features(
        signals_df,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(recent_signal_lookback_days),
    )
    labels = compute_stop_event_labels(
        featured,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        require_full_horizon=bool(require_full_horizon),
    )
    if labels.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "breakout_days": int(breakout_days),
            "recent_signal_lookback_days": int(recent_signal_lookback_days),
            "targets": {},
        }

    featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")
    merged = featured.merge(labels, on=["ticker", "signal_date", "pattern_family"], how="inner")
    if merged.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "breakout_days": int(breakout_days),
            "recent_signal_lookback_days": int(recent_signal_lookback_days),
            "targets": {},
        }

    selected_numeric_features = [
        feature for feature in (numeric_features or STOP_RISK_NUMERIC_FEATURES) if feature in merged.columns
    ]
    selected_family_levels = STOP_RISK_FAMILY_LEVELS if include_family_features else []

    targets = {
        target_name: _build_target_model(
            merged,
            target_name,
            numeric_features=selected_numeric_features,
            family_levels=selected_family_levels,
        )
        for target_name in STOP_RISK_TARGETS
    }
    return {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "breakout_days": int(breakout_days),
        "recent_signal_lookback_days": int(recent_signal_lookback_days),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "max_hold_days": int(max_hold_days),
        "feature_set_name": str(feature_set_name or "custom"),
        "numeric_features": selected_numeric_features,
        "include_family_features": bool(include_family_features),
        "require_full_horizon": bool(require_full_horizon),
        "targets": targets,
        "target_summaries": {name: _summarize_target(payload) for name, payload in targets.items()},
    }


def main() -> None:
    args = parse_args()
    prices_path = Path(args.prices)
    signals_path = Path(args.signals)
    out_path = Path(args.out)

    if not prices_path.exists():
        raise SystemExit(f"Prices file not found: {prices_path}")
    if not signals_path.exists():
        raise SystemExit(f"Signals file not found: {signals_path}")

    prices = pd.read_csv(prices_path, parse_dates=["Date"])
    signals = pd.read_csv(signals_path)
    if "signal_date" in signals.columns:
        signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
    if args.train_end_date and "signal_date" in signals.columns:
        train_end_date = pd.to_datetime(args.train_end_date, errors="coerce")
        if pd.isna(train_end_date):
            raise SystemExit(f"Invalid --train-end-date: {args.train_end_date}")
        signals = signals.loc[pd.to_datetime(signals["signal_date"], errors="coerce") <= train_end_date].copy()

    payload = compute_stop_risk_model(
        signals,
        prices,
        breakout_days=int(args.breakout_days),
        recent_signal_lookback_days=int(args.recent_signal_lookback_days),
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        max_hold_days=int(args.max_hold_days),
        numeric_features=list(STOP_RISK_FEATURE_SET_PRESETS[args.feature_set]),
        include_family_features=not bool(args.disable_family_features),
        feature_set_name=str(args.feature_set),
        require_full_horizon=not bool(args.allow_partial_horizon),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Signals analyzed: {payload.get('signals_analyzed', 0)}")
    print(f"Feature set: {payload.get('feature_set_name', 'custom')}")
    print(f"Family features enabled: {payload.get('include_family_features', True)}")
    print(f"Require full horizon: {payload.get('require_full_horizon', True)}")
    for target_name, summary in payload.get("target_summaries", {}).items():
        print(
            f"{target_name}: positive_rate={summary.get('positive_rate', 0.0):.4f}, "
            f"calibration_points={summary.get('calibration_points', 0)}"
        )
    print(f"Saved stop-risk model to: {out_path}")


if __name__ == "__main__":
    main()