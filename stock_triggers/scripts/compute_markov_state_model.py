from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, parse_optional_date
from stock_triggers.ui.patterns.markov import DEFAULT_MARKOV_SCORE_POLICY, MARKOV_STATE_LEVELS, build_markov_state_table

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_OUTPUT = DATA_DIR / "st_lt_signal_markov_model.json"
BENCHMARK_TICKERS = {"^NSEI"}


def _is_benchmark_ticker(ticker: str) -> bool:
    return str(ticker).strip().upper() in BENCHMARK_TICKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a Markov regime state model from historical prices")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--horizon-days", type=int, default=5, help="Trading-day horizon used for state transitions")
    parser.add_argument("--train-start-date", type=str, default="", help="Only use rows on or after this date (YYYY-MM-DD)")
    parser.add_argument("--train-end-date", type=str, default="", help="Only use rows on or before this date (YYYY-MM-DD)")
    parser.add_argument("--recency-half-life-months", type=float, default=12.0, help="Half-life in months for weighted transition counts. 0 disables weighting.")
    parser.add_argument("--min-transitions", type=int, default=25, help="Minimum weighted transitions required before using a state row")
    parser.add_argument("--enabled", action="store_true", help="Write the artifact with score_policy.enabled=true")
    return parser.parse_args()


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Prices file not found: {path}")
    prices = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "Close", "Volume"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise SystemExit(f"Missing required columns in prices file: {missing}")
    prices = prices[~prices["Ticker"].astype(str).map(_is_benchmark_ticker)].copy()
    prices.sort_values(["Ticker", "Date"], inplace=True)
    return prices


def build_transition_payload(
    prices_df: pd.DataFrame,
    *,
    horizon_days: int,
    train_start_date: pd.Timestamp | None,
    train_end_date: pd.Timestamp | None,
    recency_half_life_months: float,
    min_transitions: int,
    enabled: bool,
) -> dict:
    state_table = build_markov_state_table(prices_df)
    if state_table.empty:
        return {
            "model_version": 1,
            "computed_at": date.today().isoformat(),
            "horizon_days": int(horizon_days),
            "state_levels": list(MARKOV_STATE_LEVELS),
            "transitions": {},
            "state_transition_counts": {},
            "score_policy": {**DEFAULT_MARKOV_SCORE_POLICY, "enabled": bool(enabled), "horizon_days": int(horizon_days)},
        }

    states = state_table.copy()
    states["signal_date"] = pd.to_datetime(states["signal_date"], errors="coerce")
    states = filter_by_date_window(
        states,
        date_col="signal_date",
        start_date=train_start_date,
        end_date=train_end_date,
    )
    if states.empty:
        return {
            "model_version": 1,
            "computed_at": date.today().isoformat(),
            "horizon_days": int(horizon_days),
            "state_levels": list(MARKOV_STATE_LEVELS),
            "transitions": {},
            "state_transition_counts": {},
            "score_policy": {**DEFAULT_MARKOV_SCORE_POLICY, "enabled": bool(enabled), "horizon_days": int(horizon_days)},
        }

    states.sort_values(["ticker", "signal_date"], inplace=True)
    states["next_state"] = states.groupby("ticker", sort=False)["markov_state"].shift(-int(horizon_days))
    states = states.dropna(subset=["markov_state", "next_state", "signal_date"]).copy()
    if states.empty:
        return {
            "model_version": 1,
            "computed_at": date.today().isoformat(),
            "horizon_days": int(horizon_days),
            "state_levels": list(MARKOV_STATE_LEVELS),
            "transitions": {},
            "state_transition_counts": {},
            "score_policy": {**DEFAULT_MARKOV_SCORE_POLICY, "enabled": bool(enabled), "horizon_days": int(horizon_days)},
        }

    states = add_recency_weights(
        states,
        date_col="signal_date",
        half_life_months=float(recency_half_life_months),
        weight_col="sample_weight",
    )
    states["sample_weight"] = pd.to_numeric(states["sample_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = (
        states.groupby(["markov_state", "next_state"], sort=True)["sample_weight"]
        .sum()
        .reset_index(name="weighted_transitions")
    )

    transitions: dict[str, dict[str, float]] = {}
    state_transition_counts: dict[str, float] = {}
    for state_name in MARKOV_STATE_LEVELS:
        state_rows = grouped[grouped["markov_state"] == state_name].copy()
        total_weight = float(state_rows["weighted_transitions"].sum())
        if total_weight < float(min_transitions):
            continue
        state_transition_counts[state_name] = round(total_weight, 4)
        row: dict[str, float] = {}
        for _, rec in state_rows.iterrows():
            next_state = str(rec["next_state"])
            row[next_state] = round(float(rec["weighted_transitions"]) / total_weight, 6)
        transitions[state_name] = row

    return {
        "model_version": 1,
        "computed_at": date.today().isoformat(),
        "horizon_days": int(horizon_days),
        "state_levels": list(MARKOV_STATE_LEVELS),
        "transitions": transitions,
        "state_transition_counts": state_transition_counts,
        "score_policy": {**DEFAULT_MARKOV_SCORE_POLICY, "enabled": bool(enabled), "horizon_days": int(horizon_days)},
        "training": {
            "train_start_date": train_start_date.date().isoformat() if train_start_date is not None else None,
            "train_end_date": train_end_date.date().isoformat() if train_end_date is not None else None,
            "recency_half_life_months": float(recency_half_life_months),
            "min_transitions": int(min_transitions),
            "weighted_samples": round(float(states["sample_weight"].sum()), 4),
            "raw_samples": int(len(states)),
        },
    }


def main() -> None:
    args = parse_args()
    train_start_date = parse_optional_date(args.train_start_date, arg_name="train-start-date")
    train_end_date = parse_optional_date(args.train_end_date, arg_name="train-end-date")
    prices = load_prices(Path(args.prices))
    payload = build_transition_payload(
        prices,
        horizon_days=int(args.horizon_days),
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        recency_half_life_months=float(args.recency_half_life_months),
        min_transitions=int(args.min_transitions),
        enabled=bool(args.enabled),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    n_states = len(payload.get("transitions", {})) if isinstance(payload.get("transitions"), dict) else 0
    print(f"Markov model written to {out_path} ({n_states} state rows)")


if __name__ == "__main__":
    main()