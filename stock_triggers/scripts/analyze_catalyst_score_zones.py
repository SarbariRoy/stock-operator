"""Analyze catalyst score outcomes using Backtesting Lab out-of-sample methodology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.evaluate_stop_risk_walk_forward import (
    CANDIDATE_SPECS,
    _compute_labels,
    _load_signals,
    evaluate_candidate,
)
from stock_triggers.ui.patterns.catalyst_ui import CATALYST_MODES, filter_signals_by_catalyst_mode

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_OUTPUT = DATA_DIR / "catalyst_zone_analysis.csv"
DEFAULT_TRACKER_OUTPUT = DATA_DIR / "catalyst_zone_tracker.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze catalyst score outcomes with Backtesting Lab methodology.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tracker-output", type=str, default=str(DEFAULT_TRACKER_OUTPUT))
    parser.add_argument("--evaluation-mode", choices=["walk-forward", "holdout"], default="walk-forward")
    parser.add_argument("--train-end-date", type=str, default="", help="YYYY-MM-DD, used for holdout mode")
    parser.add_argument("--eval-hold-days", type=int, default=30)
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=9.0)
    parser.add_argument("--capital-per-trade", type=float, default=10000.0)
    parser.add_argument("--min-score", type=float, default=90.0)
    parser.add_argument("--max-days-held", type=int, default=60)
    parser.add_argument("--catalyst-mode", choices=list(CATALYST_MODES.keys()), default="baseline")
    parser.add_argument("--lockout-days", type=int, default=7)
    return parser.parse_args()


def _parse_train_end_date(value: str, evaluation_mode: str) -> pd.Timestamp | None:
    if str(evaluation_mode) != "holdout":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise SystemExit("--train-end-date (YYYY-MM-DD) is required when --evaluation-mode=holdout")
    return parsed


def _filter_lab_signals_for_evaluation_window(signals_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty or predictions_df.empty:
        return signals_df.iloc[0:0].copy()

    merge_keys = ["ticker", "signal_date", "pattern_family"]
    base_signals = signals_df.copy()
    if "pattern_family" not in base_signals.columns:
        base_signals["pattern_family"] = "A"
    if "pattern" in signals_df.columns and "pattern" in predictions_df.columns:
        merge_keys.append("pattern")

    predictions_view = predictions_df.copy()
    predictions_view["signal_date"] = pd.to_datetime(predictions_view["signal_date"], errors="coerce")
    predictions_view = predictions_view.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()

    prediction_columns = merge_keys + [
        col
        for col in ["month", "signal_stop_risk", "signal_reliability_score", "stop_before_target"]
        if col in predictions_view.columns
    ]
    prediction_keys = predictions_view[prediction_columns].drop_duplicates(subset=merge_keys)

    scoped = base_signals.copy()
    scoped["signal_date"] = pd.to_datetime(scoped["signal_date"], errors="coerce")
    scoped = scoped.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()
    scoped = scoped.merge(prediction_keys, on=merge_keys, how="inner")
    return scoped


def _stop_exit_allowed(signal_date: pd.Timestamp, bar_date: pd.Timestamp, *, lockout_days: int) -> bool:
    sig_dt = pd.to_datetime(signal_date)
    bar_dt = pd.to_datetime(bar_date)
    return (bar_dt - sig_dt).days > int(lockout_days)


def build_signal_tracker(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    capital_per_trade: float,
    lockout_days: int,
) -> pd.DataFrame:
    if signals_df.empty or prices_df.empty:
        return pd.DataFrame()

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker = str(sig["ticker"])
        sig_date = pd.to_datetime(sig["signal_date"])
        entry_price = float(sig["entry_price"])
        stop_price_sig = float(sig.get("stop_price", entry_price * (1.0 - stop_pct / 100.0)))
        target_price = entry_price * (1.0 + target_pct / 100.0)
        stop_price_calc = stop_price_sig
        hold_to_target_only = bool(sig.get("hold_to_target_only", False))

        qty = int(float(capital_per_trade) // entry_price) if entry_price > 0 else 0
        if qty == 0:
            continue
        invested = round(qty * entry_price, 2)

        future = prices[(prices["Ticker"] == ticker) & (prices["Date"] > sig_date)].sort_values("Date")

        status = "Holding"
        exit_date = None
        exit_price = None
        latest_close = entry_price
        for _, bar in future.iterrows():
            close = float(bar["Close"])
            high = float(bar["High"])
            latest_close = close
            if high >= target_price:
                status = "Target Hit ✅"
                exit_date = bar["Date"]
                exit_price = target_price
                break
            if (
                not hold_to_target_only
                and _stop_exit_allowed(sig_date, bar["Date"], lockout_days=int(lockout_days))
                and close <= stop_price_calc
            ):
                status = "Stop Hit 🛑"
                exit_date = bar["Date"]
                exit_price = stop_price_calc
                break

        current_val = round(qty * exit_price, 2) if exit_price is not None else round(qty * latest_close, 2)
        pnl = round(current_val - invested, 2)
        return_pct = round(((current_val / invested) - 1.0) * 100.0, 2) if invested > 0 else 0.0

        if exit_date is not None:
            days_held = (pd.to_datetime(exit_date) - sig_date).days
        elif not future.empty:
            days_held = (future["Date"].max() - sig_date).days
        else:
            days_held = 0

        rows.append(
            {
                "signal_date": sig_date.date().isoformat(),
                "ticker": ticker.replace(".NS", ""),
                "pattern": str(sig.get("pattern", "")),
                "pattern_family": str(sig.get("pattern_family", "")),
                "entry_price": round(entry_price, 2),
                "qty": qty,
                "invested": invested,
                "target_price": round(target_price, 2),
                "stop_price": round(stop_price_calc, 2),
                "latest_close": round(latest_close, 2),
                "current_value": current_val,
                "pnl": pnl,
                "return_pct": return_pct,
                "days_held": days_held,
                "exit_date": (
                    exit_date.date().isoformat()
                    if exit_date is not None and hasattr(exit_date, "date")
                    else (str(exit_date)[:10] if exit_date else "-")
                ),
                "status": status,
                "signal_score": round(float(sig["signal_score"]), 1) if pd.notna(sig.get("signal_score")) else None,
                "hold_to_target_only": hold_to_target_only,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)
    return out


def summarize_signal_tracker(view: pd.DataFrame) -> dict[str, float | int]:
    if view.empty:
        return {
            "n_total": 0,
            "n_target": 0,
            "n_stop": 0,
            "n_holding": 0,
            "total_invested": 0.0,
            "total_current": 0.0,
            "total_pnl": 0.0,
            "overall_return": 0.0,
            "closed_invested": 0.0,
            "closed_current": 0.0,
            "closed_pnl": 0.0,
            "closed_return": 0.0,
            "win_rate": 0.0,
        }

    n_total = len(view)
    n_target = int((view["status"] == "Target Hit ✅").sum())
    n_stop = int((view["status"] == "Stop Hit 🛑").sum())
    n_holding = int((view["status"] == "Holding").sum())
    total_invested = float(view["invested"].sum())
    total_current = float(view["current_value"].sum())
    total_pnl = float(view["pnl"].sum())
    overall_return = ((total_current / total_invested) - 1.0) * 100.0 if total_invested > 0 else 0.0

    closed_view = view[view["status"].isin(["Target Hit ✅", "Stop Hit 🛑"])].copy()
    closed_invested = float(closed_view["invested"].sum()) if not closed_view.empty else 0.0
    closed_current = float(closed_view["current_value"].sum()) if not closed_view.empty else 0.0
    closed_pnl = float(closed_view["pnl"].sum()) if not closed_view.empty else 0.0
    closed_return = ((closed_current / closed_invested) - 1.0) * 100.0 if closed_invested > 0 else 0.0
    win_rate = (n_target / (n_target + n_stop) * 100.0) if (n_target + n_stop) > 0 else 0.0

    return {
        "n_total": n_total,
        "n_target": n_target,
        "n_stop": n_stop,
        "n_holding": n_holding,
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "overall_return": overall_return,
        "closed_invested": closed_invested,
        "closed_current": closed_current,
        "closed_pnl": closed_pnl,
        "closed_return": closed_return,
        "win_rate": win_rate,
    }


def run_scope_evaluation(
    *,
    signals_path: Path,
    prices_path: Path,
    evaluation_mode: str,
    train_end_date: pd.Timestamp | None,
    eval_hold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    if not signals_path.exists():
        raise SystemExit(f"Signals file not found: {signals_path}")
    if not prices_path.exists():
        raise SystemExit(f"Prices file not found: {prices_path}")

    prices_df = pd.read_csv(prices_path, parse_dates=["Date"])
    if prices_df.empty:
        raise SystemExit(f"Prices file is empty: {prices_path}")

    signals_df = _load_signals(signals_path, prices_df, max_hold_days=int(eval_hold_days))
    labels_df = _compute_labels(
        signals_df,
        prices_df,
        target_pct=6.0,
        stop_pct=7.0,
        max_hold_days=int(eval_hold_days),
    )
    summary, monthly_df, predictions_df = evaluate_candidate(
        "scores_only",
        CANDIDATE_SPECS["scores_only"],
        signals_df,
        labels_df,
        prices_df,
        target_pct=6.0,
        stop_pct=7.0,
        max_hold_days=int(eval_hold_days),
        breakout_days=40,
        recent_signal_lookback_days=5,
        min_train_rows=250,
        tail_quantile=0.2,
        evaluation_mode=str(evaluation_mode),
        train_end_date=train_end_date,
        recency_half_life_months=3.0,
    )
    return signals_df, prices_df, summary, monthly_df, predictions_df


def main() -> None:
    args = parse_args()

    signals_path = Path(args.signals)
    prices_path = Path(args.prices)
    train_end_date = _parse_train_end_date(args.train_end_date, args.evaluation_mode)

    base_signals, prices_df, eval_summary, _, predictions_df = run_scope_evaluation(
        signals_path=signals_path,
        prices_path=prices_path,
        evaluation_mode=str(args.evaluation_mode),
        train_end_date=train_end_date,
        eval_hold_days=int(args.eval_hold_days),
    )

    scoped_signals = _filter_lab_signals_for_evaluation_window(base_signals, predictions_df)
    mode_signals = filter_signals_by_catalyst_mode(scoped_signals, str(args.catalyst_mode))
    score_series = pd.to_numeric(mode_signals.get("signal_score"), errors="coerce")
    score_signals = mode_signals.loc[score_series >= float(args.min_score)].copy()

    tracker_df = build_signal_tracker(
        score_signals,
        prices_df,
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        capital_per_trade=float(args.capital_per_trade),
        lockout_days=int(args.lockout_days),
    )
    tracker_view = tracker_df.loc[pd.to_numeric(tracker_df.get("days_held"), errors="coerce") <= int(args.max_days_held)].copy()

    summary = summarize_signal_tracker(tracker_view)
    avg_trade_return_pct = float(pd.to_numeric(tracker_view.get("return_pct"), errors="coerce").mean()) if not tracker_view.empty else 0.0
    median_trade_return_pct = float(pd.to_numeric(tracker_view.get("return_pct"), errors="coerce").median()) if not tracker_view.empty else 0.0

    output_summary = {
        **summary,
        "avg_trade_return_pct": avg_trade_return_pct,
        "median_trade_return_pct": median_trade_return_pct,
        "evaluation_mode": str(args.evaluation_mode),
        "train_end_date": train_end_date.date().isoformat() if train_end_date is not None else None,
        "eval_hold_days": int(args.eval_hold_days),
        "target_pct": float(args.target_pct),
        "stop_pct": float(args.stop_pct),
        "capital_per_trade": float(args.capital_per_trade),
        "min_score": float(args.min_score),
        "max_days_held": int(args.max_days_held),
        "catalyst_mode": str(args.catalyst_mode),
        "lockout_days": int(args.lockout_days),
        "oos_rows": int(eval_summary.get("oos_rows", 0) or 0),
        "scoped_rows": int(len(scoped_signals)),
        "post_mode_rows": int(len(mode_signals)),
        "post_score_rows": int(len(score_signals)),
        "tracker_rows": int(len(tracker_view)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([output_summary]).to_csv(output_path, index=False)

    tracker_output_path = Path(args.tracker_output)
    tracker_output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_view.to_csv(tracker_output_path, index=False)

    print(
        " ".join(
            [
                f"oos_rows={int(eval_summary.get('oos_rows', 0) or 0)}",
                f"scoped_rows={len(scoped_signals)}",
                f"post_mode_rows={len(mode_signals)}",
                f"post_score_rows={len(score_signals)}",
                f"tracker_rows={len(tracker_view)}",
                f"overall_return={float(summary['overall_return']):.2f}",
                f"closed_return={float(summary['closed_return']):.2f}",
                f"avg_trade_return_pct={avg_trade_return_pct:.2f}",
            ]
        )
    )


if __name__ == "__main__":
    main()