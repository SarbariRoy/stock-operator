"""Local prep runner for stock operator data and scores.

Runs the same core data steps as the daily pipeline for local usage:
1) Refresh prices
2) Generate Pattern A triggers (with signal_score)
3) Generate stock health scores
4) Print a compact verification summary
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
SCRIPTS_DIR = ROOT / "stock_triggers" / "scripts"
LT_SCRIPTS_DIR = SCRIPTS_DIR / "long_term"

PRICES_CSV = DATA_DIR / "st_lt_prices_eod.csv"
SIGNALS_CSV = DATA_DIR / "lt_signals_pattern_a.csv"
STOCK_SCORES_CSV = DATA_DIR / "stock_scores.csv"
UNIVERSE_FILE = DATA_DIR / "universe_tickers.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local prep pipeline (prices + scores)")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip price refresh step")
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Backfill Pattern A trigger generation across all dates",
    )
    parser.add_argument("--days", type=int, default=365, help="Lookback days for price refresh")
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.8,
        help="Pause seconds between refresh requests",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="Brilliant",
        help="User agent for price refresh",
    )
    return parser.parse_args()


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n==> {label}")
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Step failed ({label}) with exit code {res.returncode}")


def verify_outputs() -> None:
    if not PRICES_CSV.exists():
        raise SystemExit(f"Missing prices file: {PRICES_CSV}")
    if not SIGNALS_CSV.exists():
        raise SystemExit(f"Missing signals file: {SIGNALS_CSV}")
    if not STOCK_SCORES_CSV.exists():
        raise SystemExit(f"Missing stock scores file: {STOCK_SCORES_CSV}")

    prices = pd.read_csv(PRICES_CSV)
    signals = pd.read_csv(SIGNALS_CSV)
    scores = pd.read_csv(STOCK_SCORES_CSV)

    latest_price_date = prices["Date"].max() if "Date" in prices.columns and not prices.empty else None
    latest_signal_date = (
        signals["signal_date"].max() if "signal_date" in signals.columns and not signals.empty else None
    )

    signal_non_na = 0
    signal_min = None
    signal_max = None
    if "signal_score" in signals.columns:
        sv = pd.to_numeric(signals["signal_score"], errors="coerce")
        signal_non_na = int(sv.notna().sum())
        if sv.notna().any():
            signal_min = float(sv.min())
            signal_max = float(sv.max())

    stock_score_non_na = 0
    stock_score_min = None
    stock_score_max = None
    if "score" in scores.columns:
        mv = pd.to_numeric(scores["score"], errors="coerce")
        stock_score_non_na = int(mv.notna().sum())
        if mv.notna().any():
            stock_score_min = float(mv.min())
            stock_score_max = float(mv.max())

    print("\n==> Verification")
    print("latest_price_date:", latest_price_date)
    print("latest_signal_date:", latest_signal_date)
    print("signal_score_col:", "signal_score" in signals.columns)
    print("signal_score_non_na:", signal_non_na)
    print("signal_score_min:", signal_min)
    print("signal_score_max:", signal_max)
    print("stock_score_col:", "score" in scores.columns)
    print("stock_score_non_na:", stock_score_non_na)
    print("stock_score_min:", stock_score_min)
    print("stock_score_max:", stock_score_max)

    if "signal_score" not in signals.columns:
        raise SystemExit("signal_score column missing in lt_signals_pattern_a.csv")
    if "score" not in scores.columns:
        raise SystemExit("score column missing in stock_scores.csv")


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_refresh:
        refresh_cmd = [
            py,
            str(SCRIPTS_DIR / "update_prices_yf.py"),
            "--user-agent",
            args.user_agent,
            "--days",
            str(args.days),
            "--pause-seconds",
            str(args.pause_seconds),
            "--overwrite",
            "--universe-file",
            str(UNIVERSE_FILE),
        ]
        run_step("Refresh prices", refresh_cmd)
    else:
        print("\n==> Refresh prices")
        print("Skipped (--skip-refresh)")

    trigger_cmd = [py, str(LT_SCRIPTS_DIR / "generate_lt_signals.py")]
    if args.backfill_history:
        trigger_cmd.append("--backfill-history")
    run_step("Generate Pattern A triggers", trigger_cmd)

    run_step(
        "Generate stock health scores",
        [py, str(SCRIPTS_DIR / "generate_stock_scores.py")],
    )

    verify_outputs()
    print("\nLocal prep completed successfully.")


if __name__ == "__main__":
    main()
