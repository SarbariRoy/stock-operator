"""Phase 2: Batch enrich existing signal history with catalyst features.

Usage:
  python enrich_existing_signals.py [--signals path] [--external-factors path] [--event-calendar path] [--output path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.ui.patterns.catalyst_enrichment import enrich_signals_with_catalysts, load_external_factors, load_event_calendar

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"
DEFAULT_EXTERNAL_FACTORS = DATA_DIR / "external_factors.csv"
DEFAULT_EVENT_CALENDAR = DATA_DIR / "event_calendar.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch enrich signal history with catalyst features.")
    p.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS), help="Path to signals CSV.")
    p.add_argument("--external-factors", type=str, default=str(DEFAULT_EXTERNAL_FACTORS), help="Path to external factors CSV.")
    p.add_argument("--event-calendar", type=str, default=str(DEFAULT_EVENT_CALENDAR), help="Path to event calendar CSV.")
    p.add_argument("--output", type=str, default=None, help="Output path (default: overwrite signals).")
    p.add_argument("--skip-market-regimes", action="store_true", help="Skip market regime computation.")
    p.add_argument("--skip-event-windows", action="store_true", help="Skip event window computation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load data.
    try:
        signals = pd.read_csv(args.signals, parse_dates=["signal_date"])
    except FileNotFoundError as ex:
        raise SystemExit(f"Signal file not found: {ex}")

    external_factors = load_external_factors(Path(args.external_factors))
    event_calendar = load_event_calendar(Path(args.event_calendar))

    print(f"Loaded {len(signals)} signals from {args.signals}")
    print(f"Loaded {len(external_factors)} market-factor rows from {args.external_factors}")
    print(f"Loaded {len(event_calendar)} event rows from {args.event_calendar}")

    # Enrich with catalysts.
    enriched = enrich_signals_with_catalysts(
        signals,
        external_factors=external_factors,
        event_calendar=event_calendar,
        include_market_regimes=not args.skip_market_regimes,
        include_event_windows=not args.skip_event_windows,
    )

    # Save output.
    output_path = Path(args.output or args.signals)
    enriched.to_csv(output_path, index=False)
    print(f"\n✓ Enriched signals saved to {output_path}")
    print(f"  Added {len(enriched.columns) - len(signals.columns)} new catalyst columns")


if __name__ == "__main__":
    main()
