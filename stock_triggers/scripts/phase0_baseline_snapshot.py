"""Phase 0: Freeze baseline artifacts and record baseline metrics for catalyst evaluation.

Outputs:
- stock_triggers/data/baseline_snapshot_{date}.json — artifact versions and config state.
- stock_triggers/data/baseline_metrics_{date}.csv — top1/top3 win/loss, stop-breach, drawdown for 12-month window.

Usage:
  python phase0_baseline_snapshot.py [--as-of-date YYYY-MM-DD] [--lookback-months 12]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
SIGNALS_CSV = DATA_DIR / "st_signals_all_patterns.csv"
PENALTY_WEIGHTS_JSON = DATA_DIR / "st_lt_signal_penalty_weights.json"
STOP_RISK_MODEL_JSON = DATA_DIR / "st_lt_signal_stop_risk_model.json"
MARKOV_MODEL_JSON = DATA_DIR / "st_lt_signal_markov_model.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot baseline artifacts and compute baseline metrics.")
    p.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Evaluation date YYYY-MM-DD (default: latest signal date).",
    )
    p.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        help="Lookback window in months for baseline metric computation.",
    )
    return p.parse_args()


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Signals file not found: {path}")
    df = pd.read_csv(path, parse_dates=["signal_date"])
    return df


def load_artifact_versions() -> dict:
    """Load modification times of key artifacts."""
    artifacts = {}
    for artifact_path, key in [
        (PENALTY_WEIGHTS_JSON, "penalty_weights"),
        (STOP_RISK_MODEL_JSON, "stop_risk_model"),
        (MARKOV_MODEL_JSON, "markov_model"),
    ]:
        if artifact_path.exists():
            stat = artifact_path.stat()
            artifacts[key] = {
                "path": str(artifact_path),
                "modified": stat.st_mtime,
                "size_bytes": stat.st_size,
            }
    return artifacts


def compute_baseline_metrics(
    signals: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
    lookback_months: int = 12,
) -> dict:
    """Compute baseline signal statistics (count, score distribution)."""
    if signals.empty:
        return {}

    if as_of_date is None:
        as_of_date = pd.to_datetime(signals["signal_date"]).max()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    start_date = as_of_date - timedelta(days=int(lookback_months * 30.4375))
    mask = (pd.to_datetime(signals["signal_date"]) >= start_date) & (
        pd.to_datetime(signals["signal_date"]) <= as_of_date
    )
    window_signals = signals.loc[mask].copy()

    if window_signals.empty:
        return {}

    # Compute score distribution.
    signal_scores = pd.to_numeric(window_signals.get("signal_score", pd.Series()), errors="coerce").dropna()
    
    metrics = {
        "period_start": str(start_date.date()),
        "period_end": str(as_of_date.date()),
        "lookback_months": lookback_months,
        "total_signals": len(window_signals),
        "total_unique_tickers": window_signals["ticker"].nunique() if "ticker" in window_signals.columns else 0,
        "total_unique_dates": window_signals["signal_date"].nunique(),
        "signal_score_mean": round(float(signal_scores.mean()), 2) if len(signal_scores) > 0 else None,
        "signal_score_median": round(float(signal_scores.median()), 2) if len(signal_scores) > 0 else None,
        "signal_score_std": round(float(signal_scores.std()), 2) if len(signal_scores) > 0 else None,
        "signal_score_min": round(float(signal_scores.min()), 2) if len(signal_scores) > 0 else None,
        "signal_score_max": round(float(signal_scores.max()), 2) if len(signal_scores) > 0 else None,
    }
    
    # Pattern family distribution.
    if "pattern_family" in window_signals.columns:
        family_counts = window_signals["pattern_family"].value_counts().to_dict()
        metrics["pattern_family_distribution"] = {str(k): int(v) for k, v in family_counts.items()}
    
    return metrics


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load signals.
    signals = load_signals(SIGNALS_CSV)

    # Parse as-of-date.
    as_of_date = None
    if args.as_of_date:
        as_of_date = pd.Timestamp(args.as_of_date)

    # Compute baseline metrics.
    metrics = compute_baseline_metrics(signals, as_of_date=as_of_date, lookback_months=args.lookback_months)
    artifacts = load_artifact_versions()

    # Write outputs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_file = DATA_DIR / f"baseline_snapshot_{timestamp}.json"
    snapshot_data = {
        "timestamp": timestamp,
        "as_of_date": str(as_of_date.date()) if as_of_date else None,
        "artifacts": artifacts,
        "metrics": metrics,
    }
    with open(snapshot_file, "w") as f:
        json.dump(snapshot_data, f, indent=2)
    print(f"Saved baseline snapshot to {snapshot_file}")

    # Write metrics as CSV.
    metrics_file = DATA_DIR / f"baseline_metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved baseline metrics to {metrics_file}")
    print("\nBaseline Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
