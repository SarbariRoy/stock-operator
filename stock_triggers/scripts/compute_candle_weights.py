"""Compute candle-shape enhancer weights from historical signal outcomes.

Runs weekly (or on-demand) to analyze which candlestick patterns
(Doji, Hammer, Morning Star, Bullish Engulfing, Bullish Harami,
Piercing Line, Piercing Variant, Inverted Hammer, Bullish Belt Hold, Three White Soldiers) have a positive
win-rate edge when present at signal dates.

Output: stock_triggers/data/candle_weights.json
  {
    "doji": 2.5,
    "hammer": 1.0,
        "confirmed_hammer_a": 2.0,
    "morning_star": 3.5,
    "engulfing": 2.0,
    "harami": 1.5,
    "piercing_line": 1.0,
    "piercing_variant": 1.0,
    "inverted_hammer": 1.0,
    "belt_hold": 1.0,
    "three_white_soldiers": 2.0,
    "computed_at": "2026-03-28",
    "total_signals": 395,
    "details": { ... per-pattern stats ... }
  }

Usage:
    python stock_triggers/scripts/compute_candle_weights.py
    python stock_triggers/scripts/compute_candle_weights.py --target-pct 6 --stop-pct 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_pattern_a.csv"
DEFAULT_OUTPUT = DATA_DIR / "candle_weights.json"

from stock_triggers.ui.enhancers import (  # noqa: E402
    bullish_belt_hold,
    bullish_engulfing,
    bullish_harami,
    dragonfly_doji,
    hammer,
    inverted_hammer,
    morning_star,
    piercing_line,
    piercing_variant,
    three_white_soldiers,
)

CHECKS = [
    ("doji", dragonfly_doji.check),
    ("hammer", hammer.check),
    ("confirmed_hammer_a", None),
    ("morning_star", morning_star.check),
    ("engulfing", bullish_engulfing.check),
    ("harami", bullish_harami.check),
    ("piercing_line", piercing_line.check),
    ("piercing_variant", piercing_variant.check),
    ("inverted_hammer", inverted_hammer.check),
    ("belt_hold", bullish_belt_hold.check),
    ("three_white_soldiers", three_white_soldiers.check),
]

COMPARISON_CHECKS = [
    ("hammer_legacy", hammer.check_basic),
]


def _pattern_stats(
    df: pd.DataFrame,
    *,
    name: str,
    baseline_wr: float,
    total: int,
    min_samples: int,
    scale: float,
    max_weight: float,
) -> dict:
    with_pat = df[df[name] == True]  # noqa: E712
    without_pat = df[df[name] == False]  # noqa: E712
    n_with = len(with_pat)
    n_without = len(without_pat)

    if n_with < min_samples:
        return {
            "weight": 0.0,
            "details": {
                "count": n_with,
                "skipped": True,
                "reason": f"< {min_samples} samples",
            },
            "summary": {
                "count": n_with,
                "win_rate_with": None,
                "loss_rate_with": None,
                "win_rate_without": None,
                "loss_rate_without": None,
                "edge_pp": None,
                "weight": 0.0,
            },
        }

    wr_with = (with_pat["outcome"] == "win").mean()
    lr_with = (with_pat["outcome"] == "loss").mean()
    wr_without = (without_pat["outcome"] == "win").mean() if n_without > 0 else baseline_wr
    lr_without = (without_pat["outcome"] == "loss").mean() if n_without > 0 else 0.0

    edge_pp = (wr_with - baseline_wr) * 100
    raw = max(0.0, edge_pp * scale)
    rounded = round(raw * 2) / 2
    weight = min(rounded, max_weight)

    return {
        "weight": weight,
        "details": {
            "count": n_with,
            "pct_of_signals": round(n_with / total * 100, 1),
            "win_rate_with": round(wr_with * 100, 1),
            "loss_rate_with": round(lr_with * 100, 1),
            "win_rate_without": round(wr_without * 100, 1),
            "loss_rate_without": round(lr_without * 100, 1),
            "edge_pp": round(edge_pp, 1),
            "weight": weight,
        },
        "summary": {
            "count": n_with,
            "win_rate_with": round(wr_with * 100, 1),
            "loss_rate_with": round(lr_with * 100, 1),
            "win_rate_without": round(wr_without * 100, 1),
            "loss_rate_without": round(lr_without * 100, 1),
            "edge_pp": round(edge_pp, 1),
            "weight": weight,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute candle-shape enhancer weights from historical data")
    p.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    p.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--target-pct", type=float, default=6.0, help="Target %% for win classification")
    p.add_argument("--stop-pct", type=float, default=7.0, help="Stop %% for loss classification")
    p.add_argument("--max-hold-days", type=int, default=30, help="Max trading days to track forward")
    p.add_argument("--min-samples", type=int, default=3, help="Min pattern occurrences to assign weight")
    p.add_argument("--scale", type=float, default=0.5, help="Multiplier: weight = edge_pp * scale")
    p.add_argument("--max-weight", type=float, default=10.0, help="Cap per-pattern weight")
    return p.parse_args()


def compute_weights(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float = 6.0,
    stop_pct: float = 7.0,
    max_hold_days: int = 30,
    min_samples: int = 3,
    scale: float = 0.5,
    max_weight: float = 10.0,
) -> dict:
    """Analyze every historical signal, check candle patterns, track outcome."""

    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    grouped = {str(t): g.sort_values("Date") for t, g in prices_df.groupby("Ticker", sort=False)}

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig["ticker"])
        sd = pd.to_datetime(sig["signal_date"])
        entry = float(sig["entry_price"]) if pd.notna(sig.get("entry_price")) else None
        if entry is None or entry <= 0:
            continue
        t_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
        g = grouped.get(t_ns)
        if g is None:
            continue
        g_to = g[g["Date"] <= sd]
        if g_to.empty:
            continue

        pat_flags = {name: fn(g_to, t_ns) for name, fn in CHECKS + COMPARISON_CHECKS if fn is not None}
        pat_flags["confirmed_hammer_a"] = bool(
            pat_flags.get("hammer")
            and str(sig.get("pattern_family", "")).strip().upper() == "A"
        )

        future = g[g["Date"] > sd].head(max_hold_days)
        tp = entry * (1 + target_pct / 100)
        sp = entry * (1 - stop_pct / 100)
        outcome = "hold"
        for _, bar in future.iterrows():
            if float(bar["High"]) >= tp:
                outcome = "win"
                break
            if float(bar["Low"]) <= sp:
                outcome = "loss"
                break

        row = {"ticker": ticker, "signal_date": str(sd.date()), "outcome": outcome}
        row.update(pat_flags)
        rows.append(row)

    if not rows:
        return {
            "doji": 0.0, "hammer": 0.0, "confirmed_hammer_a": 0.0, "morning_star": 0.0, "engulfing": 0.0, "harami": 0.0, "piercing_line": 0.0,
            "piercing_variant": 0.0,
            "inverted_hammer": 0.0, "belt_hold": 0.0, "three_white_soldiers": 0.0,
            "computed_at": date.today().isoformat(),
            "total_signals": 0,
            "details": {},
        }

    df = pd.DataFrame(rows)
    total = len(df)
    n_win = int((df["outcome"] == "win").sum())
    n_loss = int((df["outcome"] == "loss").sum())
    n_hold = int((df["outcome"] == "hold").sum())
    baseline_wr = n_win / total if total > 0 else 0.0

    weights: dict[str, float] = {}
    details: dict[str, dict] = {}
    comparison_details: dict[str, dict] = {}
    comparison_summary: dict[str, dict] = {}

    for name, _ in CHECKS:
        stats = _pattern_stats(
            df,
            name=name,
            baseline_wr=baseline_wr,
            total=total,
            min_samples=min_samples,
            scale=scale,
            max_weight=max_weight,
        )
        weights[name] = stats["weight"]
        details[name] = stats["details"]

    for name, _ in COMPARISON_CHECKS:
        stats = _pattern_stats(
            df,
            name=name,
            baseline_wr=baseline_wr,
            total=total,
            min_samples=min_samples,
            scale=scale,
            max_weight=max_weight,
        )
        comparison_details[name] = stats["details"]
        comparison_summary[name] = stats["summary"]

    legacy = comparison_summary.get("hammer_legacy", {})
    confirmed = {
        "count": details.get("hammer", {}).get("count"),
        "win_rate_with": details.get("hammer", {}).get("win_rate_with"),
        "loss_rate_with": details.get("hammer", {}).get("loss_rate_with"),
        "edge_pp": details.get("hammer", {}).get("edge_pp"),
        "weight": details.get("hammer", {}).get("weight"),
    }
    hammer_vs_legacy = {
        "legacy_count": legacy.get("count"),
        "confirmed_count": confirmed.get("count"),
        "sample_change": (
            int(confirmed["count"]) - int(legacy["count"])
            if legacy.get("count") is not None and confirmed.get("count") is not None
            else None
        ),
        "legacy_win_rate_with": legacy.get("win_rate_with"),
        "confirmed_win_rate_with": confirmed.get("win_rate_with"),
        "win_rate_lift_pp": (
            round(float(confirmed["win_rate_with"]) - float(legacy["win_rate_with"]), 1)
            if legacy.get("win_rate_with") is not None and confirmed.get("win_rate_with") is not None
            else None
        ),
        "legacy_edge_pp": legacy.get("edge_pp"),
        "confirmed_edge_pp": confirmed.get("edge_pp"),
        "edge_lift_pp": (
            round(float(confirmed["edge_pp"]) - float(legacy["edge_pp"]), 1)
            if legacy.get("edge_pp") is not None and confirmed.get("edge_pp") is not None
            else None
        ),
        "legacy_weight_if_used": legacy.get("weight"),
        "confirmed_weight": confirmed.get("weight"),
    }

    result = {
        **weights,
        "computed_at": date.today().isoformat(),
        "total_signals": total,
        "baseline_win_rate": round(baseline_wr * 100, 1),
        "outcomes": {"win": n_win, "loss": n_loss, "hold": n_hold},
        "details": details,
        "comparison_details": comparison_details,
        "comparisons": {
            "hammer_vs_legacy": hammer_vs_legacy,
        },
    }
    return result


def main() -> None:
    args = parse_args()

    prices_path = Path(args.prices)
    signals_path = Path(args.signals)
    out_path = Path(args.out)

    if not prices_path.exists():
        print(f"ERROR: Prices file not found: {prices_path}")
        sys.exit(1)
    if not signals_path.exists():
        print(f"ERROR: Signals file not found: {signals_path}")
        sys.exit(1)

    print(f"Loading prices from {prices_path} ...")
    prices = pd.read_csv(prices_path, parse_dates=["Date"])
    print(f"  {len(prices):,} rows, {prices['Ticker'].nunique()} tickers")

    print(f"Loading signals from {signals_path} ...")
    signals = pd.read_csv(signals_path)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    print(f"  {len(signals):,} signals")

    print(f"\nComputing weights (target={args.target_pct}%, stop={args.stop_pct}%, hold={args.max_hold_days}d) ...")
    result = compute_weights(
        signals, prices,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        max_hold_days=args.max_hold_days,
        min_samples=args.min_samples,
        scale=args.scale,
        max_weight=args.max_weight,
    )

    print(f"\n{'='*50}")
    print(f"Baseline win rate: {result['baseline_win_rate']}%")
    print(f"Total signals analyzed: {result['total_signals']}")
    print(f"Outcomes: {result['outcomes']}")
    print(f"\nDerived weights:")
    for name in ("doji", "hammer", "confirmed_hammer_a", "morning_star", "engulfing", "harami", "piercing_line", "piercing_variant", "inverted_hammer", "belt_hold", "three_white_soldiers"):
        w = result[name]
        d = result["details"].get(name, {})
        edge = d.get("edge_pp", "n/a")
        count = d.get("count", 0)
        wr = d.get("win_rate_with", "n/a")
        print(f"  {name:15s}  weight={w:5.1f}  (edge={edge}pp, n={count}, wr={wr}%)")
    hammer_comparison = result.get("comparisons", {}).get("hammer_vs_legacy", {})
    if hammer_comparison:
        print("\nHammer comparison:")
        print(
            "  legacy n={legacy_n} wr={legacy_wr}% edge={legacy_edge}pp | "
            "confirmed n={confirmed_n} wr={confirmed_wr}% edge={confirmed_edge}pp".format(
                legacy_n=hammer_comparison.get("legacy_count", "n/a"),
                legacy_wr=hammer_comparison.get("legacy_win_rate_with", "n/a"),
                legacy_edge=hammer_comparison.get("legacy_edge_pp", "n/a"),
                confirmed_n=hammer_comparison.get("confirmed_count", "n/a"),
                confirmed_wr=hammer_comparison.get("confirmed_win_rate_with", "n/a"),
                confirmed_edge=hammer_comparison.get("confirmed_edge_pp", "n/a"),
            )
        )
        print(
            f"  lift: win_rate={hammer_comparison.get('win_rate_lift_pp', 'n/a')}pp, "
            f"edge={hammer_comparison.get('edge_lift_pp', 'n/a')}pp"
        )
    print(f"{'='*50}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
