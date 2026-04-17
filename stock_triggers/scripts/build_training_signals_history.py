"""Build a shared training artifact for downstream signal-model training.

The artifact is intentionally additive in this staged rollout: current trainers
still read the existing CSV inputs, but this file materializes the overlapping
features and forward labels they repeatedly recompute today.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.compute_signal_stop_risk_model import compute_stop_event_labels
from stock_triggers.scripts.generate_signals_all_patterns import load_prices
from stock_triggers.training_utils import filter_by_date_window, parse_optional_date
from stock_triggers.ui.patterns.penalties import compute_signal_penalty_features, get_recent_signal_lookback_days, load_signal_penalty_weights
from stock_triggers.ui.patterns.stop_risk import prepare_stop_risk_features

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_OUTPUT = DATA_DIR / "training_signals_history.csv"
RECENT_SIGNAL_LOOKBACK_CANDIDATES = (5, 10, 20, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shared training artifact from historical signals and prices")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--max-hold-days", type=int, default=30)
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument(
        "--recent-signal-lookback-days",
        type=int,
        default=0,
        help="0 = use the current learned penalty payload setting",
    )
    parser.add_argument(
        "--train-start-date",
        type=str,
        default="",
        help="Only keep rows on or after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="",
        help="Only keep rows on or before this date (YYYY-MM-DD)",
    )
    return parser.parse_args()


def _resolve_recent_signal_lookback_days(explicit_value: int) -> int:
    if int(explicit_value) > 0:
        return int(explicit_value)
    return int(get_recent_signal_lookback_days(load_signal_penalty_weights()))


def _resolve_price_history(grouped: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in grouped:
        return grouped[clean]
    if clean.endswith(".NS"):
        return grouped.get(clean[:-3])
    return grouped.get(clean + ".NS")


def _compute_forward_outcomes(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices.groupby("Ticker", sort=False)}

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig.get("ticker", "")).strip()
        pattern_family = str(sig.get("pattern_family", "")).strip().upper()
        signal_date = pd.to_datetime(sig.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(sig.get("entry_price"), errors="coerce")
        if not ticker or not pattern_family or pd.isna(signal_date) or pd.isna(entry_price) or float(entry_price) <= 0:
            continue

        hist = _resolve_price_history(grouped, ticker)
        if hist is None:
            continue

        future = hist[hist["Date"] > signal_date].head(int(max_hold_days)).copy()
        bars_available_forward = int(len(future))
        target_price = float(entry_price) * (1.0 + float(target_pct) / 100.0)
        effective_stop_price = pd.to_numeric(sig.get("stop_price"), errors="coerce")
        if pd.isna(effective_stop_price) or float(effective_stop_price) <= 0 or float(effective_stop_price) >= float(entry_price):
            effective_stop_price = float(entry_price) * (1.0 - float(stop_pct) / 100.0)
        else:
            effective_stop_price = float(effective_stop_price)

        first_target_hit_day: int | None = None
        first_stop_hit_day: int | None = None
        outcome = "hold"
        for day_number, (_, bar) in enumerate(future.iterrows(), start=1):
            high_value = pd.to_numeric(bar.get("High"), errors="coerce")
            low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
            if pd.notna(high_value) and float(high_value) >= target_price:
                first_target_hit_day = day_number
                outcome = "win"
                break
            if pd.notna(low_value) and float(low_value) <= float(effective_stop_price):
                first_stop_hit_day = day_number
                outcome = "loss"
                break

        if first_target_hit_day is None:
            for day_number, (_, bar) in enumerate(future.iterrows(), start=1):
                high_value = pd.to_numeric(bar.get("High"), errors="coerce")
                if pd.notna(high_value) and float(high_value) >= target_price:
                    first_target_hit_day = day_number
                    break
        if first_stop_hit_day is None:
            for day_number, (_, bar) in enumerate(future.iterrows(), start=1):
                low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
                if pd.notna(low_value) and float(low_value) <= float(effective_stop_price):
                    first_stop_hit_day = day_number
                    break

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date.date().isoformat(),
                "pattern_family": pattern_family,
                "outcome_30d": outcome,
                "win_flag": int(outcome == "win"),
                "loss_flag": int(outcome == "loss"),
                "hold_flag": int(outcome == "hold"),
                "bars_available_forward": bars_available_forward,
                "target_price": round(float(target_price), 6),
                "effective_stop_price": round(float(effective_stop_price), 6),
                "first_target_hit_day": first_target_hit_day,
                "first_stop_hit_day": first_stop_hit_day,
            }
        )

    return pd.DataFrame(rows)


def build_training_signals_history(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    recent_signal_lookback_days: int,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
) -> pd.DataFrame:
    featured = prepare_stop_risk_features(
        signals_df,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(recent_signal_lookback_days),
    )
    featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")

    stop_labels = compute_stop_event_labels(
        featured,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        require_full_horizon=False,
    )
    forward_outcomes = _compute_forward_outcomes(
        featured,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
    )

    training = featured.merge(
        stop_labels,
        on=["ticker", "signal_date", "pattern_family"],
        how="left",
    )
    training = training.merge(
        forward_outcomes,
        on=["ticker", "signal_date", "pattern_family"],
        how="left",
    )

    signal_score = pd.to_numeric(training.get("signal_score"), errors="coerce")
    score_penalty_total = pd.to_numeric(training.get("score_penalty_total"), errors="coerce").fillna(0.0)
    training["signal_score_pre_penalty"] = (signal_score - score_penalty_total).round(4)
    training["month"] = pd.to_datetime(training["signal_date"], errors="coerce").dt.to_period("M").astype("string")

    for lookback_days in RECENT_SIGNAL_LOOKBACK_CANDIDATES:
        lookback_featured = compute_signal_penalty_features(
            signals_df,
            prices_df,
            breakout_days=int(breakout_days),
            recent_signal_lookback_days=int(lookback_days),
        )
        lookback_featured["signal_date"] = pd.to_datetime(lookback_featured["signal_date"], errors="coerce").dt.date.astype("string")
        lookback_featured = lookback_featured[
            ["ticker", "signal_date", "pattern_family", "feature_recent_signal_count"]
        ].copy()
        lookback_featured.rename(
            columns={"feature_recent_signal_count": f"feature_recent_signal_count_{int(lookback_days)}d"},
            inplace=True,
        )
        training = training.merge(
            lookback_featured,
            on=["ticker", "signal_date", "pattern_family"],
            how="left",
        )

    ordered_columns: list[str] = []
    for column in [
        "ticker",
        "signal_date",
        "month",
        "pattern",
        "pattern_family",
        "entry_price",
        "stop_pct",
        "stop_price",
        "target_price",
        "effective_stop_price",
        "bars_available_forward",
        "first_target_hit_day",
        "first_stop_hit_day",
        "outcome_30d",
        "win_flag",
        "loss_flag",
        "hold_flag",
        "stop_before_target",
        "stop_within_5d",
        "gap_through_stop",
        "mae_exceeds_stop",
        "signal_score_pre_penalty",
        "signal_score",
        "score_trend",
        "score_setup",
        "score_volume",
        "score_rsi",
        "score_risk",
        "score_pattern",
        "ma_slope_bonus",
        "pattern_bonus",
        "feature_recent_signal_count",
        "feature_close_vs_prev_high_pct",
        "feature_close_vs_sma50_pct",
        "feature_gap_pct",
        "feature_range_vs_atr",
        "feature_gap_sequence_risk",
        "feature_exhaustion_risk",
        "score_penalty_crowding",
        "score_penalty_extension",
        "score_penalty_gap_shock",
        "score_penalty_total",
        "consensus_count",
        "feature_recent_signal_count_5d",
        "feature_recent_signal_count_10d",
        "feature_recent_signal_count_20d",
        "feature_recent_signal_count_40d",
    ]:
        if column in training.columns and column not in ordered_columns:
            ordered_columns.append(column)

    for column in training.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    return training[ordered_columns].copy()


def _write_output(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Parquet output requested but no parquet engine is installed. "
                "Use a .csv output path or add pyarrow/fastparquet to requirements."
            ) from exc
        return
    df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    prices_path = Path(args.prices)
    signals_path = Path(args.signals)
    out_path = Path(args.out)

    if not prices_path.exists():
        raise SystemExit(f"Prices file not found: {prices_path}")
    if not signals_path.exists():
        raise SystemExit(f"Signals file not found: {signals_path}")

    prices = load_prices(prices_path)
    signals = pd.read_csv(signals_path)
    if "signal_date" in signals.columns:
        signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
    train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
    train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
    signals = filter_by_date_window(
        signals,
        date_col="signal_date",
        start_date=train_start_date,
        end_date=train_end_date,
    )

    recent_signal_lookback_days = _resolve_recent_signal_lookback_days(int(args.recent_signal_lookback_days))
    training = build_training_signals_history(
        signals,
        prices,
        breakout_days=int(args.breakout_days),
        recent_signal_lookback_days=recent_signal_lookback_days,
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        max_hold_days=int(args.max_hold_days),
    )
    _write_output(training, out_path)

    print(f"Signals loaded: {len(signals)}")
    print(f"Training rows written: {len(training)}")
    print(f"Recent-signal lookback days: {recent_signal_lookback_days}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()