"""Phase 2: Penalty model catalyst extension.

Extends penalty training to optionally include catalyst features:
- Market regimes: vix_regime_high, flow_regime_weak, energy_regime_shock
- Event windows: within_earnings_window, within_dividend_window, post_event_gap_risk, event_proximity_score

Enforces feature caps to prevent catalyst terms from overpowering pattern-family signal quality:
- Per-feature adjustment cap: 4 points
- Total catalyst adjustment cap per row: 8 points
- Fallback to current artifact if feature coverage < min threshold

Usage:
  python train_catalyst_penalties.py [--signals path] [--prices path] [--output model.json] \
    [--feature-cap 4.0] [--total-cap 8.0] [--coverage-threshold 0.85]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_OUTPUT = DATA_DIR / "signal_penalty_weights_catalyst.json"
CURRENT_PENALTY_MODEL = DATA_DIR / "signal_penalty_weights.json"


CATALYST_FEATURES = [
    "vix_regime_high",
    "flow_regime_weak",
    "energy_regime_shock",
    "within_earnings_window",
    "within_dividend_window",
    "post_event_gap_risk",
    "event_proximity_score",
]


def load_catalyst_penalties_baseline(path: Path = CURRENT_PENALTY_MODEL) -> dict:
    """Load current penalty model as fallback if catalyst training fails."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def compute_signal_outcomes(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Compute outcome (win/loss) for each signal if available."""
    signals = signals.copy()
    
    if "outcome" not in signals.columns:
        # Attempt to infer outcome from entry/exit prices or mark all as unknown.
        if "entry_price" in signals.columns and "exit_price" in signals.columns:
            signals["outcome_return"] = (
                (signals["exit_price"] - signals["entry_price"]) / signals["entry_price"]
            )
            signals["outcome"] = signals["outcome_return"].apply(lambda x: 1 if x > 0 else 0)
        else:
            signals["outcome"] = np.nan
    else:
        signals["outcome"] = signals["outcome"].map({"W": 1, "L": 0, "U": np.nan})
    
    return signals


def train_catalyst_penalty_model(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    feature_cap: float = 4.0,
    total_cap: float = 8.0,
    coverage_threshold: float = 0.85,
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Train catalyst-aware penalty weights with feature caps.
    
    Returns:
        Dict with keys:
        - "baseline_model": current penalty weights (fallback)
        - "catalyst_features": list of catalyst features used
        - "feature_weights": dict of feature -> coefficient
        - "feature_caps": dict of feature -> max adjustment
        - "total_cap": total adjustment cap per row
        - "coverage": fraction of signals with full feature coverage
        - "fallback_reason": reason for fallback if applicable
    """
    signals = compute_signal_outcomes(signals, prices)
    
    # Check coverage of catalyst features.
    feature_coverage = {}
    for feature in CATALYST_FEATURES:
        if feature in signals.columns:
            coverage = signals[feature].notna().sum() / len(signals)
            feature_coverage[feature] = coverage
    
    # Determine which features to use.
    usable_features = [f for f in feature_coverage if feature_coverage[f] >= coverage_threshold]
    
    if not usable_features:
        fallback = load_catalyst_penalties_baseline()
        return {
            "mode": "fallback",
            "reason": f"Insufficient feature coverage (threshold={coverage_threshold})",
            "baseline_model": fallback,
            "catalyst_features": [],
            "feature_weights": {},
            "feature_caps": {},
        }
    
    # Prepare training data.
    train_mask = signals["outcome"].notna()
    if train_mask.sum() < 10:
        fallback = load_catalyst_penalties_baseline()
        return {
            "mode": "fallback",
            "reason": f"Insufficient labeled outcomes ({train_mask.sum()} signals)",
            "baseline_model": fallback,
            "catalyst_features": [],
            "feature_weights": {},
            "feature_caps": {},
        }
    
    X = signals.loc[train_mask, usable_features].fillna(0).astype(float)
    y = signals.loc[train_mask, "outcome"].astype(float)
    
    # Standardize features for interpretability.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train penalized regression.
    model = Ridge(alpha=regularization)
    model.fit(X_scaled, y)
    
    # Map coefficients back to original scale and apply caps.
    feature_weights = {}
    for i, feature in enumerate(usable_features):
        raw_coef = float(model.coef_[i])
        scale_factor = scaler.scale_[i]
        adjusted_coef = raw_coef / scale_factor if scale_factor > 0 else 0
        
        # Apply per-feature cap.
        capped_coef = np.clip(adjusted_coef, -feature_cap, feature_cap)
        feature_weights[feature] = round(float(capped_coef), 2)
    
    # Validate total cap enforcement (done at inference time).
    feature_caps = {feature: feature_cap for feature in usable_features}
    
    return {
        "mode": "trained",
        "catalyst_features": usable_features,
        "feature_weights": feature_weights,
        "feature_caps": feature_caps,
        "total_cap": total_cap,
        "coverage": round(float(np.mean([feature_coverage[f] for f in usable_features])), 3),
        "baseline_model": load_catalyst_penalties_baseline(),
        "metadata": {
            "train_samples": int(train_mask.sum()),
            "regularization_alpha": regularization,
            "feature_coverage_threshold": coverage_threshold,
        },
    }


def apply_catalyst_penalties(
    signal_row: dict,
    catalyst_weights: dict[str, Any],
    *,
    current_score: float = None,
) -> tuple[float, dict]:
    """Apply catalyst adjustments to a signal score with caps.
    
    Returns:
        (adjusted_score, adjustment_details)
    """
    if catalyst_weights["mode"] != "trained":
        return (current_score or 0, {"reason": "fallback_mode"})
    
    feature_weights = catalyst_weights["feature_weights"]
    total_cap = catalyst_weights["total_cap"]
    
    adjustments = {}
    total_adjustment = 0.0
    
    for feature, weight in feature_weights.items():
        value = signal_row.get(feature, 0)
        if pd.isna(value):
            value = 0
        
        # Convert boolean to int if needed.
        if isinstance(value, bool):
            value = int(value)
        
        adjustment = float(value) * float(weight)
        adjustments[feature] = round(adjustment, 2)
        total_adjustment += adjustment
    
    # Apply total cap.
    total_adjustment = np.clip(total_adjustment, -total_cap, total_cap)
    
    adjusted_score = (current_score or 0) + total_adjustment
    
    return adjusted_score, {
        "feature_adjustments": adjustments,
        "total_adjustment": round(total_adjustment, 2),
        "adjusted_score": round(adjusted_score, 2),
    }


def main() -> None:
    import argparse
    
    p = argparse.ArgumentParser(description="Train catalyst-aware penalty model.")
    p.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS), help="Signals CSV path.")
    p.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Prices CSV path.")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output model path.")
    p.add_argument("--feature-cap", type=float, default=4.0, help="Per-feature adjustment cap.")
    p.add_argument("--total-cap", type=float, default=8.0, help="Total catalyst adjustment cap.")
    p.add_argument("--coverage-threshold", type=float, default=0.85, help="Min feature coverage to use.")
    p.add_argument("--regularization", type=float, default=1.0, help="Ridge regression alpha.")
    p.add_argument("--fallback-only", action="store_true", help="Skip training; save fallback artifact.")
    args = p.parse_args()
    
    if args.fallback_only:
        fallback = load_catalyst_penalties_baseline()
        output = {"mode": "fallback", "baseline_model": fallback}
    else:
        signals = pd.read_csv(args.signals, parse_dates=["signal_date"])
        prices = pd.read_csv(args.prices, parse_dates=["Date"])
        
        output = train_catalyst_penalty_model(
            signals,
            prices,
            feature_cap=args.feature_cap,
            total_cap=args.total_cap,
            coverage_threshold=args.coverage_threshold,
            regularization=args.regularization,
        )
    
    # Save artifact.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Catalyst penalty model saved to {args.output}")
    print(f"Mode: {output['mode']}")
    if output["mode"] == "trained":
        print(f"Features trained: {output['catalyst_features']}")
        print(f"Average coverage: {output['coverage']}")


if __name__ == "__main__":
    main()
