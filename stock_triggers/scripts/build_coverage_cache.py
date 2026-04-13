from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.coverage_cache import DEFAULT_CACHE_PKL, DEFAULT_PRICES_CSV, DEFAULT_SIGNALS_CSV, build_and_save_default_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build persisted default Coverage cache for the Streamlit app")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES_CSV), help="Input prices CSV path")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS_CSV), help="Input all-pattern signals CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_CACHE_PKL), help="Output pickle path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_and_save_default_cache(
        prices_path=Path(args.prices),
        signals_path=Path(args.signals),
        cache_path=Path(args.out),
    )
    df = payload.get("df")
    meta = payload.get("meta") or {}
    print(f"Saved coverage cache: {args.out}")
    print(
        "rows="
        f"{len(df) if hasattr(df, '__len__') else 0} "
        f"target={meta.get('target_return_pct')}% "
        f"forward_days={meta.get('forward_days')} "
        f"threshold={meta.get('score_threshold')} "
        f"pending={meta.get('pending_count')}"
    )


if __name__ == "__main__":
    main()