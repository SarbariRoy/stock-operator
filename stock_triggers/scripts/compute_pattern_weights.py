"""Compute data-driven pattern-family score weights from historical signal outcomes.

Runs on-demand (or from the daily pipeline) to analyze which pattern families
(A-G) deserve more or less weight inside the total propensity score.

Output: stock_triggers/data/st_lt_pattern_weights.json
  {
        "A": 26.3,
        "B": 7.8,
        "C": 15.6,
    ...
    "computed_at": "2026-03-30",
    "total_signals": 812,
    "details": { ... per-family stats ... }
  }
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

from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, get_sample_weight_series, parse_optional_date, weighted_mean

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"
DEFAULT_TRAINING_DATA = DATA_DIR / "st_lt_training_signals_history.csv"
DEFAULT_OUTPUT = DATA_DIR / "st_lt_pattern_weights.json"
PATTERN_FAMILIES = ("A", "B", "C", "D", "E", "F", "G")


def clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pattern-family weights from historical signal outcomes")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument(
        "--training-data",
        type=str,
        default="",
        help="Optional shared training artifact with precomputed outcomes",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-pct", type=float, default=6.0, help="Target %% for win classification")
    parser.add_argument("--stop-pct", type=float, default=7.0, help="Fallback stop %% if a row stop is missing")
    parser.add_argument("--max-hold-days", type=int, default=30, help="Max trading days to track forward")
    parser.add_argument("--min-samples", type=int, default=5, help="Min historical samples to assign a weight")
    parser.add_argument("--scale", type=float, default=4.0, help="Multiplier used to convert adjusted edge into a 0-100 pattern score")
    parser.add_argument("--max-weight", type=float, default=30.0, help="Cap per-family contribution inside the total score")
    parser.add_argument("--confidence-samples", type=int, default=100, help="Samples needed before full confidence is given to a family edge")
    parser.add_argument("--train-start-date", type=str, default="", help="Only use rows on or after this date (YYYY-MM-DD)")
    parser.add_argument("--train-end-date", type=str, default="", help="Only use rows on or before this date (YYYY-MM-DD)")
    parser.add_argument("--recency-half-life-months", type=float, default=0.0, help="Half-life in months for recency weighting. 0 disables weighting.")
    return parser.parse_args()


def _summarize_outcomes(
    outcomes_df: pd.DataFrame,
    *,
    min_samples: int,
    scale: float,
    max_weight: float,
    confidence_samples: int,
) -> dict:
    if outcomes_df.empty:
        return {
            **{family: 0.0 for family in PATTERN_FAMILIES},
            "computed_at": date.today().isoformat(),
            "total_signals": 0,
            "details": {},
        }

    df = outcomes_df.copy()
    df["pattern_family"] = df["pattern_family"].astype(str).str.strip().str.upper()
    df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
    df = df[df["pattern_family"].isin(PATTERN_FAMILIES)].copy()
    df = df[df["outcome"].isin({"win", "loss", "hold"})].copy()
    if df.empty:
        return {
            **{family: 0.0 for family in PATTERN_FAMILIES},
            "computed_at": date.today().isoformat(),
            "total_signals": 0,
            "details": {},
        }

    total = len(df)
    sample_weight = get_sample_weight_series(df)
    n_win = int((df["outcome"] == "win").sum())
    n_loss = int((df["outcome"] == "loss").sum())
    n_hold = int((df["outcome"] == "hold").sum())
    baseline_wr = weighted_mean((df["outcome"] == "win").astype(float), sample_weight)

    weights: dict[str, float] = {}
    details: dict[str, dict] = {}
    for family in PATTERN_FAMILIES:
        fam_df = df[df["pattern_family"] == family].copy()
        fam_count = len(fam_df)

        if fam_count < int(min_samples):
            weights[family] = 0.0
            details[family] = {
                "count": fam_count,
                "score_pattern": 0.0,
                "skipped": True,
                "reason": f"< {int(min_samples)} samples",
            }
            continue

        fam_weights = get_sample_weight_series(fam_df)
        wr = weighted_mean((fam_df["outcome"] == "win").astype(float), fam_weights)
        lr = weighted_mean((fam_df["outcome"] == "loss").astype(float), fam_weights)
        edge_pp = (wr - baseline_wr) * 100.0
        confidence = min(1.0, fam_count / max(1.0, float(confidence_samples)))
        weighted_edge_pp = edge_pp * confidence
        score_pattern = round(clip_score(50.0 + weighted_edge_pp * float(scale)), 1)
        weight = round(min(float(max_weight), (score_pattern / 100.0) * float(max_weight)), 1)
        weights[family] = weight
        details[family] = {
            "count": fam_count,
            "pct_of_signals": round(fam_count / total * 100.0, 1),
            "win_rate_with": round(wr * 100.0, 1),
            "loss_rate_with": round(lr * 100.0, 1),
            "edge_pp": round(edge_pp, 1),
            "confidence": round(confidence, 2),
            "weighted_edge_pp": round(weighted_edge_pp, 1),
            "score_pattern": score_pattern,
            "weight": weights[family],
        }

    return {
        **weights,
        "computed_at": date.today().isoformat(),
        "total_signals": total,
        "baseline_win_rate": round(baseline_wr * 100.0, 1),
        "outcomes": {"win": n_win, "loss": n_loss, "hold": n_hold},
        "details": details,
    }


def _load_training_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_weights_from_training_data(
    training_df: pd.DataFrame,
    *,
    min_samples: int = 5,
    scale: float = 4.0,
    max_weight: float = 30.0,
    confidence_samples: int = 100,
) -> dict:
    outcome_column = "outcome_30d" if "outcome_30d" in training_df.columns else "outcome"
    required_columns = {"pattern_family", outcome_column}
    missing = sorted(required_columns - set(training_df.columns))
    if missing:
        raise SystemExit(f"Training data missing required columns: {missing}")

    columns = ["pattern_family", outcome_column]
    if "signal_date" in training_df.columns:
        columns.append("signal_date")
    if "sample_weight" in training_df.columns:
        columns.append("sample_weight")
    outcomes = training_df[columns].copy()
    outcomes.rename(columns={outcome_column: "outcome"}, inplace=True)
    return _summarize_outcomes(
        outcomes,
        min_samples=int(min_samples),
        scale=float(scale),
        max_weight=float(max_weight),
        confidence_samples=int(confidence_samples),
    )


def compute_weights(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float = 6.0,
    stop_pct: float = 7.0,
    max_hold_days: int = 30,
    min_samples: int = 5,
    scale: float = 4.0,
    max_weight: float = 30.0,
    confidence_samples: int = 100,
) -> dict:
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices_df.groupby("Ticker", sort=False)}

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig.get("ticker", "")).strip()
        family = str(sig.get("pattern_family", "")).strip().upper()
        if not ticker or family not in PATTERN_FAMILIES:
            continue

        signal_date = pd.to_datetime(sig.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(sig.get("entry_price"), errors="coerce")
        if pd.isna(signal_date) or pd.isna(entry_price) or float(entry_price) <= 0:
            continue

        ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
        hist = grouped.get(ticker_ns)
        if hist is None:
            hist = grouped.get(ticker)
        if hist is None:
            continue

        future = hist[hist["Date"] > signal_date].head(int(max_hold_days))
        target_price = float(entry_price) * (1.0 + float(target_pct) / 100.0)
        row_stop_price = pd.to_numeric(sig.get("stop_price"), errors="coerce")
        if pd.isna(row_stop_price) or float(row_stop_price) <= 0 or float(row_stop_price) >= float(entry_price):
            row_stop_price = float(entry_price) * (1.0 - float(stop_pct) / 100.0)

        outcome = "hold"
        for _, bar in future.iterrows():
            if float(bar["High"]) >= target_price:
                outcome = "win"
                break
            if float(bar["Low"]) <= float(row_stop_price):
                outcome = "loss"
                break

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date.date().isoformat(),
                "pattern_family": family,
                "outcome": outcome,
                "sample_weight": float(sig.get("sample_weight", 1.0)) if pd.notna(sig.get("sample_weight", 1.0)) else 1.0,
            }
        )

    return _summarize_outcomes(
        pd.DataFrame(rows),
        min_samples=int(min_samples),
        scale=float(scale),
        max_weight=float(max_weight),
        confidence_samples=int(confidence_samples),
    )


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
        if float(args.recency_half_life_months) > 0:
            training = add_recency_weights(training, date_col="signal_date", half_life_months=float(args.recency_half_life_months))
        print("\nComputing pattern-family weights from precomputed outcomes ...")
        result = compute_weights_from_training_data(
            training,
            min_samples=args.min_samples,
            scale=args.scale,
            max_weight=args.max_weight,
            confidence_samples=args.confidence_samples,
        )
    else:
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
        if "signal_date" in signals.columns:
            signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
        train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
        train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
        signals = filter_by_date_window(signals, date_col="signal_date", start_date=train_start_date, end_date=train_end_date)
        if float(args.recency_half_life_months) > 0:
            signals = add_recency_weights(signals, date_col="signal_date", half_life_months=float(args.recency_half_life_months))
        print(f"  {len(signals):,} signals")

        print(f"\nComputing pattern-family weights (target={args.target_pct}%, stop={args.stop_pct}%, hold={args.max_hold_days}d) ...")
        result = compute_weights(
            signals,
            prices,
            target_pct=args.target_pct,
            stop_pct=args.stop_pct,
            max_hold_days=args.max_hold_days,
            min_samples=args.min_samples,
            scale=args.scale,
            max_weight=args.max_weight,
            confidence_samples=args.confidence_samples,
        )

    print(f"\n{'=' * 50}")
    print(f"Baseline win rate: {result.get('baseline_win_rate', 0.0)}%")
    print(f"Total signals analyzed: {result.get('total_signals', 0)}")
    print(f"Outcomes: {result.get('outcomes', {})}")
    print("\nDerived pattern weights:")
    for family in PATTERN_FAMILIES:
        detail = result.get("details", {}).get(family, {})
        print(
            f"  {family}: weight={float(result.get(family, 0.0)):4.1f}/30 "
            f"(score={detail.get('score_pattern', 'n/a')}/100, edge={detail.get('edge_pp', 'n/a')}pp, n={detail.get('count', 0)}, wr={detail.get('win_rate_with', 'n/a')}%)"
        )
    print(f"{'=' * 50}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()