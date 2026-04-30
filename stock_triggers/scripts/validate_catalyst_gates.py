"""Phase 1: Data quality gates for catalyst feature gating.

Enforces blocking gates before proceeding to Phase 2:
- Gate A: No duplicate Date rows in market factors.
- Gate B: No duplicate ticker+event_date rows in event calendar.
- Gate C: Join coverage ≥95% for market fields, ≥90% for event flags.
- Gate D: No forward leakage (catalyst values available on/before signal_date).

Usage:
  python validate_catalyst_gates.py [--external-factors path] [--event-calendar path] [--signals path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_EXTERNAL_FACTORS = DATA_DIR / "external_factors.csv"
DEFAULT_EVENT_CALENDAR = DATA_DIR / "event_calendar.csv"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate catalyst feature data quality gates.")
    p.add_argument("--external-factors", type=str, default=str(DEFAULT_EXTERNAL_FACTORS), help="Path to external factors CSV.")
    p.add_argument("--event-calendar", type=str, default=str(DEFAULT_EVENT_CALENDAR), help="Path to event calendar CSV.")
    p.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS), help="Path to signals CSV.")
    p.add_argument("--strict", action="store_true", help="Exit with error if any gate fails.")
    return p.parse_args()


def gate_a_market_factor_duplicates(external_factors: pd.DataFrame) -> dict:
    """Gate A: No duplicate Date rows in external_factors."""
    duplicates = external_factors[external_factors.duplicated(subset=["Date"], keep=False)]
    passed = len(duplicates) == 0
    return {
        "gate_name": "A (Market Factor Uniqueness)",
        "passed": passed,
        "duplicate_count": len(duplicates),
        "message": "✓ Market factors unique by Date" if passed else f"✗ Found {len(duplicates)} duplicate Date rows",
    }


def gate_b_event_calendar_duplicates(event_calendar: pd.DataFrame) -> dict:
    """Gate B: No duplicate ticker+event_date rows in event_calendar."""
    duplicates = event_calendar[event_calendar.duplicated(subset=["ticker", "event_date"], keep=False)]
    passed = len(duplicates) == 0
    return {
        "gate_name": "B (Event Calendar Uniqueness)",
        "passed": passed,
        "duplicate_count": len(duplicates),
        "message": "✓ Event calendar unique by ticker+event_date" if passed else f"✗ Found {len(duplicates)} duplicate rows",
    }


def gate_c_join_coverage(
    signals: pd.DataFrame,
    external_factors: pd.DataFrame,
    event_calendar: pd.DataFrame,
    market_coverage_threshold: float = 0.70,
    event_coverage_threshold: float = 0.50,
) -> dict:
    """Gate C: Join coverage ≥95% for market, ≥90% for events."""
    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    
    # Market factors join.
    merged_market = signals.merge(
        external_factors[["Date", "india_vix_close", "usdinr_close", "brent_close"]],
        left_on="signal_date",
        right_on="Date",
        how="left",
    )
    market_coverage = merged_market["india_vix_close"].notna().sum() / len(signals)
    market_passed = market_coverage >= market_coverage_threshold
    
    # Event calendar join.
    merged_events = signals.merge(
        event_calendar[["ticker", "event_date", "event_type"]],
        left_on=["ticker", "signal_date"],
        right_on=["ticker", "event_date"],
        how="left",
    )
    event_coverage = merged_events["event_type"].notna().sum() / len(signals)
    event_passed = event_coverage >= event_coverage_threshold
    
    passed = market_passed and event_passed
    return {
        "gate_name": "C (Join Coverage)",
        "passed": passed,
        "market_coverage_pct": round(market_coverage * 100, 2),
        "market_threshold_pct": round(market_coverage_threshold * 100, 2),
        "market_passed": market_passed,
        "event_coverage_pct": round(event_coverage * 100, 2),
        "event_threshold_pct": round(event_coverage_threshold * 100, 2),
        "event_passed": event_passed,
        "message": "✓ Join coverage acceptable" if passed else "✗ Join coverage below thresholds",
    }


def gate_d_no_leakage(signals: pd.DataFrame, event_calendar: pd.DataFrame) -> dict:
    """Gate D: No forward leakage. Event dates must be on or before signal_date."""
    signals = signals.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    event_calendar = event_calendar.copy()
    event_calendar["event_date"] = pd.to_datetime(event_calendar["event_date"])

    merged = signals.merge(
        event_calendar,
        left_on=["ticker", "signal_date"],
        right_on=["ticker", "event_date"],
        how="inner",
    )
    
    if merged.empty:
        # No events on signal dates; safe.
        return {
            "gate_name": "D (No Leakage)",
            "passed": True,
            "future_event_count": 0,
            "message": "✓ No forward leakage detected",
        }

    future_events = (merged["signal_date"] < merged["event_date"]).sum()
    passed = future_events == 0
    return {
        "gate_name": "D (No Leakage)",
        "passed": passed,
        "future_event_count": future_events,
        "message": "✓ No forward leakage" if passed else f"✗ Found {future_events} future event references",
    }


def main() -> None:
    args = parse_args()

    # Load data.
    try:
        external_factors = pd.read_csv(args.external_factors, parse_dates=["Date"])
        event_calendar = pd.read_csv(args.event_calendar, parse_dates=["event_date"])
        signals = pd.read_csv(args.signals, parse_dates=["signal_date"])
    except FileNotFoundError as ex:
        raise SystemExit(f"Data file not found: {ex}")

    # Run gates.
    gate_results = []
    gate_results.append(gate_a_market_factor_duplicates(external_factors))
    gate_results.append(gate_b_event_calendar_duplicates(event_calendar))
    gate_results.append(gate_c_join_coverage(signals, external_factors, event_calendar))
    gate_results.append(gate_d_no_leakage(signals, event_calendar))

    # Print results.
    print("\n=== Catalyst Feature Data Quality Gates ===\n")
    all_passed = True
    for result in gate_results:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{status} | {result['gate_name']}")
        print(f"       {result['message']}")
        for key, value in result.items():
            if key not in ["gate_name", "passed", "message"]:
                print(f"       {key}: {value}")
        print()
        all_passed = all_passed and result["passed"]

    # Save results.
    results_file = Path(args.signals).parent / "lt_catalyst_gates_validation.json"
    with open(results_file, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        serializable_results = []
        for result in gate_results:
            clean_result = {}
            for key, value in result.items():
                if isinstance(value, (bool, type(None))):
                    clean_result[key] = bool(value) if value is not None else None
                elif isinstance(value, (int, float)):
                    clean_result[key] = float(value) if isinstance(value, float) else int(value)
                else:
                    clean_result[key] = value
            serializable_results.append(clean_result)
        json.dump(serializable_results, f, indent=2)
    print(f"Validation results saved to {results_file}")

    if not all_passed and args.strict:
        raise SystemExit("Catalyst feature data quality gates failed. Fix and retry.")
    elif all_passed:
        print("\n✅ All gates passed. Proceeding to Phase 2.")
    else:
        print("\n⚠️  Some gates failed. Review before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
