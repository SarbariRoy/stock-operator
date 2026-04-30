"""Phase 3: Comparative evaluation and acceptance gates.

Runs signals through three catalyst modes (baseline, market-only, market+events) and compares
rankings, metrics to determine if catalyst features improve predictions per acceptance criteria.

Output:
- stock_triggers/data/lt_catalyst_evaluation_results_{date}.json
- stock_triggers/data/lt_catalyst_evaluation_metrics_{date}.csv

Acceptance gates:
1. Top1 win-rate delta: ≥+1.5 percentage points vs baseline
2. Top3 win-rate delta: ≥+1.0 percentage point vs baseline
3. Stop-breach delta: no worse than +0.5 percentage points vs baseline
4. Max drawdown delta: no worse than +3% relative vs baseline
5. Signal retention: ≥70% vs baseline

Usage:
  python evaluate_catalyst_modes.py [--signals path] [--test-window-days 90] [--strict]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.ui.patterns.catalyst_ui import filter_signals_by_catalyst_mode

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"


def compute_ranking_metrics(
    signals: pd.DataFrame,
    *,
    label: str = "all",
) -> dict:
    """Compute top1/top3 win rates and other metrics for a signal set."""
    if signals.empty:
        return {
            "mode": label,
            "total_signals": 0,
            "top1_wins": 0,
            "top1_total": 0,
            "top1_win_pct": None,
            "top3_wins": 0,
            "top3_total": 0,
            "top3_win_pct": None,
            "stop_breach_pct": None,
            "max_drawdown_pct": None,
        }

    # Assume outcome column exists (W, L, or 1, 0).
    if "outcome" not in signals.columns:
        outcomes_computed = False
    else:
        outcomes_computed = signals["outcome"].notna().sum() > 0

    if not outcomes_computed:
        return {
            "mode": label,
            "total_signals": len(signals),
            "top1_wins": None,
            "top1_total": None,
            "top1_win_pct": None,
            "top3_wins": None,
            "top3_total": None,
            "top3_win_pct": None,
            "stop_breach_pct": None,
            "max_drawdown_pct": None,
            "note": "No outcome column; metrics unavailable",
        }

    # Map outcome to binary.
    outcome_map = {"W": 1, "L": 0}
    signals_eval = signals.copy()
    if signals_eval["outcome"].dtype == "object":
        signals_eval["outcome_binary"] = signals_eval["outcome"].map(outcome_map)
    else:
        signals_eval["outcome_binary"] = pd.to_numeric(signals_eval["outcome"], errors="coerce")

    # Top1: highest signal_score.
    top1 = signals_eval.nlargest(1, "signal_score")
    top1_wins = (top1["outcome_binary"] == 1).sum() if not top1.empty else 0
    top1_total = len(top1)
    top1_win_pct = (top1_wins / top1_total * 100) if top1_total > 0 else None

    # Top3: 3 highest signal_scores.
    top3 = signals_eval.nlargest(3, "signal_score")
    top3_wins = (top3["outcome_binary"] == 1).sum() if not top3.empty else 0
    top3_total = len(top3)
    top3_win_pct = (top3_wins / top3_total * 100) if top3_total > 0 else None

    # Stop breach: count signals where exit_price <= stop_price.
    stop_breach_pct = None
    if "exit_price" in signals_eval.columns and "stop_price" in signals_eval.columns:
        breaches = (signals_eval["exit_price"] <= signals_eval["stop_price"]).sum()
        stop_breach_pct = (breaches / len(signals_eval) * 100) if len(signals_eval) > 0 else 0

    # Max drawdown.
    max_drawdown_pct = None
    if "exit_price" in signals_eval.columns:
        prices = pd.to_numeric(signals_eval["exit_price"], errors="coerce").dropna()
        if len(prices) > 0:
            cummax = prices.cummax()
            drawdowns = (prices - cummax) / cummax * 100
            max_drawdown_pct = drawdowns.min()

    return {
        "mode": label,
        "total_signals": len(signals_eval),
        "top1_wins": int(top1_wins),
        "top1_total": top1_total,
        "top1_win_pct": round(top1_win_pct, 2) if top1_win_pct is not None else None,
        "top3_wins": int(top3_wins),
        "top3_total": top3_total,
        "top3_win_pct": round(top3_win_pct, 2) if top3_win_pct is not None else None,
        "stop_breach_pct": round(stop_breach_pct, 2) if stop_breach_pct is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
    }


def evaluate_acceptance_gates(
    baseline_metrics: dict,
    market_only_metrics: dict,
    market_and_events_metrics: dict,
    *,
    top1_delta_threshold: float = 1.5,
    top3_delta_threshold: float = 1.0,
    stop_breach_tolerance: float = 0.5,
    drawdown_tolerance_pct: float = 3.0,
    retention_threshold: float = 0.70,
) -> dict:
    """Check all acceptance gates and return pass/fail status."""
    
    gates = {}
    all_passed = True

    # Gate 1: Top1 win-rate delta.
    baseline_top1 = baseline_metrics.get("top1_win_pct")
    market_top1 = market_and_events_metrics.get("top1_win_pct")
    if baseline_top1 is not None and market_top1 is not None:
        delta = market_top1 - baseline_top1
        passed = delta >= top1_delta_threshold
        gates["gate_1_top1_delta"] = {
            "passed": passed,
            "baseline": baseline_top1,
            "market_and_events": market_top1,
            "delta": round(delta, 2),
            "threshold": top1_delta_threshold,
            "message": f"✓ Top1 delta +{delta:.2f}pp" if passed else f"✗ Top1 delta +{delta:.2f}pp (need ≥+{top1_delta_threshold}pp)",
        }
        all_passed = all_passed and passed
    else:
        gates["gate_1_top1_delta"] = {"passed": False, "message": "Unable to compute (missing metrics)"}

    # Gate 2: Top3 win-rate delta.
    baseline_top3 = baseline_metrics.get("top3_win_pct")
    market_top3 = market_and_events_metrics.get("top3_win_pct")
    if baseline_top3 is not None and market_top3 is not None:
        delta = market_top3 - baseline_top3
        passed = delta >= top3_delta_threshold
        gates["gate_2_top3_delta"] = {
            "passed": passed,
            "baseline": baseline_top3,
            "market_and_events": market_top3,
            "delta": round(delta, 2),
            "threshold": top3_delta_threshold,
            "message": f"✓ Top3 delta +{delta:.2f}pp" if passed else f"✗ Top3 delta +{delta:.2f}pp (need ≥+{top3_delta_threshold}pp)",
        }
        all_passed = all_passed and passed
    else:
        gates["gate_2_top3_delta"] = {"passed": False, "message": "Unable to compute (missing metrics)"}

    # Gate 3: Stop-breach deterioration.
    baseline_breach = baseline_metrics.get("stop_breach_pct")
    market_breach = market_and_events_metrics.get("stop_breach_pct")
    if baseline_breach is not None and market_breach is not None:
        delta = market_breach - baseline_breach
        passed = delta <= stop_breach_tolerance
        gates["gate_3_stop_breach"] = {
            "passed": passed,
            "baseline": baseline_breach,
            "market_and_events": market_breach,
            "delta": round(delta, 2),
            "tolerance": stop_breach_tolerance,
            "message": f"✓ Stop-breach delta {delta:+.2f}pp" if passed else f"✗ Stop-breach delta {delta:+.2f}pp (tolerance ≤+{stop_breach_tolerance}pp)",
        }
        all_passed = all_passed and passed
    else:
        gates["gate_3_stop_breach"] = {"passed": False, "message": "Unable to compute (missing metrics)"}

    # Gate 4: Max drawdown deterioration.
    baseline_dd = baseline_metrics.get("max_drawdown_pct")
    market_dd = market_and_events_metrics.get("max_drawdown_pct")
    if baseline_dd is not None and market_dd is not None and baseline_dd < 0:
        relative_delta = ((market_dd - baseline_dd) / baseline_dd * 100) if baseline_dd != 0 else 0
        passed = relative_delta <= drawdown_tolerance_pct
        gates["gate_4_drawdown"] = {
            "passed": passed,
            "baseline": baseline_dd,
            "market_and_events": market_dd,
            "relative_delta_pct": round(relative_delta, 2),
            "tolerance_pct": drawdown_tolerance_pct,
            "message": f"✓ Drawdown delta {relative_delta:+.2f}% relative" if passed else f"✗ Drawdown delta {relative_delta:+.2f}% (tolerance ≤+{drawdown_tolerance_pct}%)",
        }
        all_passed = all_passed and passed
    else:
        gates["gate_4_drawdown"] = {"passed": False, "message": "Unable to compute (missing or invalid metrics)"}

    # Gate 5: Signal retention.
    baseline_count = baseline_metrics.get("total_signals", 0)
    market_count = market_and_events_metrics.get("total_signals", 0)
    if baseline_count > 0:
        retention = market_count / baseline_count
        passed = retention >= retention_threshold
        gates["gate_5_retention"] = {
            "passed": passed,
            "baseline_count": baseline_count,
            "market_and_events_count": market_count,
            "retention_pct": round(retention * 100, 2),
            "threshold_pct": round(retention_threshold * 100, 2),
            "message": f"✓ Retention {retention*100:.1f}%" if passed else f"✗ Retention {retention*100:.1f}% (need ≥{retention_threshold*100:.0f}%)",
        }
        all_passed = all_passed and passed
    else:
        gates["gate_5_retention"] = {"passed": False, "message": "Unable to compute (baseline_count=0)"}

    return {
        "all_passed": all_passed,
        "gates": gates,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate catalyst modes against acceptance gates.")
    p.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS), help="Signals CSV path.")
    p.add_argument("--test-window-days", type=int, default=90, help="Window size for recent evaluation (days).")
    p.add_argument("--strict", action="store_true", help="Exit with error if gates fail.")
    args = p.parse_args()

    # Load signals.
    try:
        signals = pd.read_csv(args.signals, parse_dates=["signal_date"])
    except FileNotFoundError as ex:
        raise SystemExit(f"Signals file not found: {ex}")

    # Filter to recent window.
    latest_date = pd.to_datetime(signals["signal_date"]).max()
    window_start = latest_date - timedelta(days=args.test_window_days)
    signals_window = signals[pd.to_datetime(signals["signal_date"]) >= window_start].copy()

    print(f"\n=== Phase 3: Catalyst Mode Evaluation ===")
    print(f"Period: {window_start.date()} to {latest_date.date()} ({args.test_window_days} days)")
    print(f"Total signals in window: {len(signals_window)}\n")

    # Evaluate three modes.
    baseline_metrics = compute_ranking_metrics(signals_window, label="baseline")
    market_only_signals = filter_signals_by_catalyst_mode(signals_window, "market_only")
    market_only_metrics = compute_ranking_metrics(market_only_signals, label="market_only")

    market_events_signals = filter_signals_by_catalyst_mode(signals_window, "market_and_events")
    market_events_metrics = compute_ranking_metrics(market_events_signals, label="market_and_events")

    # Check acceptance gates.
    evaluation = evaluate_acceptance_gates(baseline_metrics, market_only_metrics, market_events_metrics)

    # Print results.
    print("Ranking Metrics:")
    for mode_label, metrics in [
        ("Baseline", baseline_metrics),
        ("Market-Only", market_only_metrics),
        ("Market + Events", market_events_metrics),
    ]:
        print(f"  {mode_label}:")
        print(f"    Signals: {metrics['total_signals']}")
        print(f"    Top1 win-rate: {metrics['top1_win_pct']}%")
        print(f"    Top3 win-rate: {metrics['top3_win_pct']}%")
        print(f"    Stop-breach: {metrics['stop_breach_pct']}%")
        print(f"    Max drawdown: {metrics['max_drawdown_pct']}%\n")

    print("Acceptance Gates:")
    for gate_key, gate_result in evaluation["gates"].items():
        status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
        print(f"  {status} | {gate_key}: {gate_result['message']}")

    # Save results.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(args.signals).parent / f"lt_catalyst_evaluation_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "window_start": str(window_start.date()),
                "window_end": str(latest_date.date()),
                "baseline": baseline_metrics,
                "market_only": market_only_metrics,
                "market_and_events": market_events_metrics,
                "evaluation": evaluation,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")

    if not evaluation["all_passed"] and args.strict:
        raise SystemExit("Acceptance gates failed.")


if __name__ == "__main__":
    main()
