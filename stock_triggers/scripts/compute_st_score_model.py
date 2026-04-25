"""Train a logistic regression model to predict P(hit 3% target within 7 days before hitting 3% stop)."""

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

from stock_triggers.ui.patterns.markov import (
    apply_signal_markov_model,
    build_markov_state_table,
    load_signal_markov_model,
)
from stock_triggers.ui.patterns.stop_risk import (
    _fit_isotonic_regression,
    _fit_logistic_regression,
    _sigmoid,
)
from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, parse_optional_date

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNAL_ST_SCORE_MODEL_JSON = DATA_DIR / "signal_st_score_model.json"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_TRAINING_DATA = DATA_DIR / "training_signals_history.csv"

ST_NUMERIC_FEATURES = [
    # Base signal components
    "score_trend",
    "score_setup",
    "score_volume",
    "score_rsi",
    "score_risk",
    # Penalty components (strong negative predictors for 7-day moves)
    "score_penalty_extension",
    "score_penalty_crowding",
    "score_penalty_gap_shock",
    # Raw intraday / structural features
    "feature_gap_pct",
    "feature_exhaustion_risk",
    "feature_close_vs_prev_high_pct",
    "feature_range_vs_atr",
    # Markov chain state probabilities (available in live signals)
    "markov_p_continuation",
    "markov_p_adverse",
    # Context and market regime
    "consensus_count",
    "feature_recent_signal_count",
    "regime_median_ret_20d_pct",
    "regime_pct_above_sma50",
]
ST_FAMILY_LEVELS = ["A", "B", "C", "D", "E", "F", "G"]
ST_TARGET = "st_hit_target_7d"

# 3% target, 2% stop, 7 days horizon
ST_TARGET_PCT = 3.0
ST_STOP_PCT = 2.0
ST_HOLD_DAYS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a calibrated ST score model to predict 3% target hit within 7 days"
    )
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument(
        "--training-data",
        type=str,
        default="",
        help="Optional shared training artifact with precomputed features and ST labels",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNAL_ST_SCORE_MODEL_JSON))
    parser.add_argument(
        "--target-pct",
        type=float,
        default=ST_TARGET_PCT,
        help="Target profit % (default 3.0)",
    )
    parser.add_argument(
        "--stop-pct",
        type=float,
        default=ST_STOP_PCT,
        help="Stop loss % (default 3.0)",
    )
    parser.add_argument(
        "--hold-days",
        type=int,
        default=ST_HOLD_DAYS,
        help="Maximum hold period in days (default 7)",
    )
    parser.add_argument(
        "--train-start-date",
        type=str,
        default="",
        help="Only use signals on or after this date for training (YYYY-MM-DD)",
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
        help="Allow rows with fewer than hold-days of future prices when building labels",
    )
    parser.add_argument(
        "--recency-half-life-months",
        type=float,
        default=0.0,
        help="Half-life in months for recency weighting. 0 disables weighting.",
    )
    return parser.parse_args()


def _load_training_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_price_history(grouped: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in grouped:
        return grouped[clean]
    if clean.endswith(".NS"):
        return grouped.get(clean[:-3])
    return grouped.get(clean + ".NS")


def compute_st_outcome_labels(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float = 3.0,
    stop_pct: float = 3.0,
    hold_days: int = 7,
    require_full_horizon: bool = True,
) -> pd.DataFrame:
    """Compute 7-day outcome labels for ST scoring.
    
    Label: 1 if max_price >= entry_price * (1 + target_pct/100) AND never hit entry_price * (1 - stop_pct/100)
    Label: 0 if hit stop before target
    """
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

        future = hist[hist["Date"] > signal_date].head(int(hold_days)).copy()
        if future.empty:
            continue
        if bool(require_full_horizon) and len(future) < int(hold_days):
            continue

        entry_price = float(entry_price)
        target_price = entry_price * (1.0 + float(target_pct) / 100.0)
        stop_price = entry_price * (1.0 - float(stop_pct) / 100.0)

        # Check if hit target or stop
        hit_target = False
        hit_stop = False

        for _, bar in future.iterrows():
            low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
            high_value = pd.to_numeric(bar.get("High"), errors="coerce")
            
            # Check stop first (we care if target hit AND stop never hit)
            if pd.notna(low_value) and float(low_value) <= stop_price:
                hit_stop = True
                break
            if pd.notna(high_value) and float(high_value) >= target_price:
                hit_target = True
                # Don't break yet, keep checking for stop
                
        # Label: 1 if hit target AND NOT hit stop
        label = 1 if (hit_target and not hit_stop) else 0

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date.date().isoformat(),
                "pattern_family": family,
                "st_hit_target_7d": int(label),
            }
        )

    return pd.DataFrame(rows)


def compute_st_features(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    markov_model: dict | None = None,
) -> pd.DataFrame:
    """Compute ST-specific features for model training.
    
    Includes:
    - Markov state and probabilities
    - Intraday gap and volatility
    - Recent momentum (1d, 3d, 5d)
    - Crowding indicators
    """
    features = signals_df.copy()
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    
    # Initialize feature columns
    features["markov_state"] = ""
    features["markov_p_continuation"] = 0.0
    features["markov_p_adverse"] = 0.0
    features["gap_pct"] = 0.0
    features["volatility_pct"] = 0.0
    features["momentum_1d_pct"] = 0.0
    features["momentum_3d_pct"] = 0.0
    features["momentum_5d_pct"] = 0.0
    
    # Build markov state table
    markov_table = build_markov_state_table(prices)
    
    # Load markov model for probabilities
    if markov_model is None:
        markov_model = load_signal_markov_model()
    transitions = markov_model.get("transitions", {})
    
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices.groupby("Ticker", sort=False)}
    
    for idx, row in features.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        
        if pd.isna(signal_date):
            continue
            
        # Get markov state
        markov_match = markov_table[
            (markov_table["ticker"] == ticker.upper()) &
            (markov_table["signal_date"] == signal_date.date().isoformat())
        ]
        if not markov_match.empty:
            state = str(markov_match.iloc[0].get("markov_state", ""))
            features.at[idx, "markov_state"] = state
            
            # Get markov probabilities
            if state in transitions and isinstance(transitions.get(state), dict):
                transitions_from_state = transitions[state]
                p_cont = 0.0
                p_adv = 0.0
                for next_state, raw_prob in transitions_from_state.items():
                    try:
                        prob = float(raw_prob)
                    except (TypeError, ValueError):
                        continue
                    if next_state in {"constructive_trend", "fresh_breakout"}:
                        p_cont += prob
                    if next_state in {"extended_breakout", "breakdown_risk"}:
                        p_adv += prob
                features.at[idx, "markov_p_continuation"] = round(float(max(0.0, min(1.0, p_cont))), 4)
                features.at[idx, "markov_p_adverse"] = round(float(max(0.0, min(1.0, p_adv))), 4)
        
        # Get price data for signal date and after
        hist = _resolve_price_history(grouped, ticker)
        if hist is None:
            continue
        
        signal_bar = hist[hist["Date"].dt.date == signal_date.date()]
        if signal_bar.empty:
            continue
        
        signal_bar = signal_bar.iloc[0]
        open_price = pd.to_numeric(signal_bar.get("Open"), errors="coerce")
        close_price = pd.to_numeric(signal_bar.get("Close"), errors="coerce")
        high_price = pd.to_numeric(signal_bar.get("High"), errors="coerce")
        low_price = pd.to_numeric(signal_bar.get("Low"), errors="coerce")
        
        # Intraday gap %
        if pd.notna(open_price) and float(open_price) > 0:
            gap_pct = ((float(close_price) - float(open_price)) / float(open_price)) * 100
            features.at[idx, "gap_pct"] = round(gap_pct, 4)
        
        # Intraday volatility %
        if pd.notna(close_price) and float(close_price) > 0 and pd.notna(high_price) and pd.notna(low_price):
            volatility_pct = ((float(high_price) - float(low_price)) / float(close_price)) * 100
            features.at[idx, "volatility_pct"] = round(volatility_pct, 4)
        
        # Recent momentum
        future = hist[hist["Date"].dt.date > signal_date.date()].copy()
        if len(future) >= 1:
            future_close_1d = pd.to_numeric(future.iloc[0].get("Close"), errors="coerce")
            if pd.notna(close_price) and pd.notna(future_close_1d) and float(close_price) > 0:
                momentum_1d = ((float(future_close_1d) - float(close_price)) / float(close_price)) * 100
                features.at[idx, "momentum_1d_pct"] = round(momentum_1d, 4)
        
        if len(future) >= 3:
            future_close_3d = pd.to_numeric(future.iloc[2].get("Close"), errors="coerce")
            if pd.notna(close_price) and pd.notna(future_close_3d) and float(close_price) > 0:
                momentum_3d = ((float(future_close_3d) - float(close_price)) / float(close_price)) * 100
                features.at[idx, "momentum_3d_pct"] = round(momentum_3d, 4)
        
        if len(future) >= 5:
            future_close_5d = pd.to_numeric(future.iloc[4].get("Close"), errors="coerce")
            if pd.notna(close_price) and pd.notna(future_close_5d) and float(close_price) > 0:
                momentum_5d = ((float(future_close_5d) - float(close_price)) / float(close_price)) * 100
                features.at[idx, "momentum_5d_pct"] = round(momentum_5d, 4)
    
    return features


def _summarize_model(payload: dict) -> dict:
    return {
        "positive_rate": round(float(payload.get("positive_rate", 0.0)), 4),
        "calibration_points": int(len(payload.get("isotonic_upper_bounds", []))),
    }


def _build_st_score_model(
    feature_df: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    family_levels: list[str] | None = None,
) -> dict:
    """Build logistic regression model with isotonic calibration."""
    numeric_features = [
        feature for feature in (numeric_features or ST_NUMERIC_FEATURES) if feature in feature_df.columns
    ]
    family_levels = list(family_levels or ST_FAMILY_LEVELS)
    working = feature_df.copy()
    
    # Remove rows with NaN target or missing key data
    working = working.dropna(subset=["ticker", "signal_date", "pattern_family", ST_TARGET])
    if working.empty:
        return {
            "positive_rate": 0.0,
            "numeric_features": numeric_features,
            "family_levels": family_levels,
            "impute_medians": {},
            "scaler_means": {},
            "scaler_stds": {},
            "coefficients": [],
            "intercept": 0.0,
            "isotonic_upper_bounds": [],
            "isotonic_values": [],
        }
    
    weight_source = working["sample_weight"] if "sample_weight" in working.columns else pd.Series(1.0, index=working.index)
    sample_weight = pd.to_numeric(weight_source, errors="coerce").fillna(0.0).astype("float64").to_numpy()

    impute_medians: dict[str, float] = {}
    scaler_means: dict[str, float] = {}
    scaler_stds: dict[str, float] = {}
    numeric_parts: list[np.ndarray] = []

    # Normalize numeric features
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
        
        normalized = ((filled - mean_value) / std_value).to_numpy().reshape(-1, 1)
        numeric_parts.append(normalized)

    # One-hot encode pattern family
    family_series = working.get("pattern_family", pd.Series("", index=working.index)).astype(str).str.strip().str.upper()
    family_parts = [
        (family_series == family).astype("float64").to_numpy().reshape(-1, 1)
        for family in family_levels
    ]

    # Build feature matrix
    X_parts = numeric_parts + family_parts
    X = np.hstack(X_parts).astype("float64") if X_parts else np.zeros((len(working), 0), dtype="float64")
    y = pd.to_numeric(working[ST_TARGET], errors="coerce").fillna(0.0).astype("float64").to_numpy()

    # Fit logistic regression
    coefficients, intercept = _fit_logistic_regression(X, y, sample_weight=sample_weight)
    raw_probabilities = _sigmoid((X @ coefficients) + intercept)
    
    # Fit isotonic calibration
    isotonic_upper_bounds, isotonic_values = _fit_isotonic_regression(
        raw_probabilities, y, sample_weight=sample_weight
    )

    return {
        "positive_rate": round(
            float(np.average(y, weights=sample_weight)) if len(y) and float(sample_weight.sum()) > 0 else 0.0,
            10,
        ),
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


def compute_st_score_model(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float = 3.0,
    stop_pct: float = 3.0,
    hold_days: int = 7,
    numeric_features: list[str] | None = None,
    include_family_features: bool = True,
    require_full_horizon: bool = True,
) -> dict:
    """Build ST score model from live signals and prices."""
    # Compute labels
    labels = compute_st_outcome_labels(
        signals_df,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        hold_days=int(hold_days),
        require_full_horizon=bool(require_full_horizon),
    )
    
    if labels.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "hold_days": int(hold_days),
            "numeric_features": [],
            "include_family_features": bool(include_family_features),
            "model": {},
        }
    
    # Compute features
    featured = compute_st_features(signals_df, prices_df)
    featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")
    
    # Merge labels with features
    merged = featured.merge(labels, on=["ticker", "signal_date", "pattern_family"], how="inner")
    if merged.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "hold_days": int(hold_days),
            "numeric_features": [],
            "include_family_features": bool(include_family_features),
            "model": {},
        }
    
    selected_numeric_features = [
        feature for feature in (numeric_features or ST_NUMERIC_FEATURES) if feature in merged.columns
    ]
    selected_family_levels = ST_FAMILY_LEVELS if include_family_features else []
    
    model_payload = _build_st_score_model(
        merged,
        numeric_features=selected_numeric_features,
        family_levels=selected_family_levels,
    )
    
    return {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "hold_days": int(hold_days),
        "numeric_features": selected_numeric_features,
        "include_family_features": bool(include_family_features),
        "require_full_horizon": bool(require_full_horizon),
        "model": model_payload,
        "model_summary": _summarize_model(model_payload),
    }


def compute_st_score_model_from_training_data(
    training_df: pd.DataFrame,
    *,
    target_pct: float = 3.0,
    stop_pct: float = 3.0,
    hold_days: int = 7,
    numeric_features: list[str] | None = None,
    include_family_features: bool = True,
    require_full_horizon: bool = True,
) -> dict:
    """Build ST score model from pre-computed training data."""
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
            "target_pct": float(target_pct),
            "stop_pct": float(stop_pct),
            "hold_days": int(hold_days),
            "numeric_features": [],
            "include_family_features": bool(include_family_features),
            "model": {},
        }
    
    selected_numeric_features = [
        feature for feature in (numeric_features or ST_NUMERIC_FEATURES) if feature in merged.columns
    ]
    selected_family_levels = ST_FAMILY_LEVELS if include_family_features else []
    
    model_payload = _build_st_score_model(
        merged,
        numeric_features=selected_numeric_features,
        family_levels=selected_family_levels,
    )
    
    return {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "hold_days": int(hold_days),
        "numeric_features": selected_numeric_features,
        "include_family_features": bool(include_family_features),
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

    if training_data_path.exists():
        print(f"Loading training artifact from {training_data_path} ...")
        training = _load_training_data(training_data_path)
        print(f"  {len(training):,} rows")
        
        train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
        train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
        training = filter_by_date_window(training, date_col="signal_date", start_date=train_start_date, end_date=train_end_date)
        
        # If ST labels not in training data, compute from prices
        if ST_TARGET not in training.columns:
            print(f"Computing ST labels from prices...")
            if not prices_path.exists():
                raise SystemExit(f"Prices file not found: {prices_path}")
            prices = pd.read_csv(prices_path, parse_dates=["Date"])
            st_labels = compute_st_outcome_labels(
                training,
                prices,
                target_pct=float(args.target_pct),
                stop_pct=float(args.stop_pct),
                hold_days=int(args.hold_days),
            )
            training = training.merge(st_labels, on=["ticker", "signal_date", "pattern_family"], how="inner")
            print(f"  After merging labels: {len(training):,} rows")
        
        if float(args.recency_half_life_months) > 0:
            training = add_recency_weights(training, date_col="signal_date", half_life_months=float(args.recency_half_life_months))

        payload = compute_st_score_model_from_training_data(
            training,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            include_family_features=True,
            require_full_horizon=not bool(args.allow_partial_horizon),
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

        payload = compute_st_score_model(
            signals,
            prices,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            hold_days=int(args.hold_days),
            include_family_features=True,
            require_full_horizon=not bool(args.allow_partial_horizon),
        )

    # Save model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
    print(f"\nST Score Model Summary:")
    print(f"  Computed at: {payload.get('computed_at', 'N/A')}")
    print(f"  Signals analyzed: {payload.get('signals_analyzed', 0)}")
    print(f"  Target %: {payload.get('target_pct', 0)}%")
    print(f"  Stop %: {payload.get('stop_pct', 0)}%")
    print(f"  Hold days: {payload.get('hold_days', 0)}")
    
    summary = payload.get("model_summary", {})
    print(f"  Target hit rate: {summary.get('positive_rate', 0.0):.4f}")
    print(f"  Calibration points: {summary.get('calibration_points', 0)}")
    print(f"  Numeric features: {len(payload.get('numeric_features', []))}")
    print(f"  Family features: {payload.get('include_family_features', False)}")
    
    print(f"\nSaved ST score model to: {out_path}")


if __name__ == "__main__":
    main()
