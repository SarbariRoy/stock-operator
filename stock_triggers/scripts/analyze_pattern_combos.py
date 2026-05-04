"""Analyze exact pattern-family combos versus single-pattern cohorts offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.compute_signal_stop_risk_model import compute_stop_event_labels
from stock_triggers.scripts.short_term.generate_st_signals import _score_pattern_a_rows, load_pattern_weights
from stock_triggers.indicators import compute_rsi
from stock_triggers.ui.patterns import STANDARD_SIGNAL_COLS
from stock_triggers.ui.patterns import pattern_a, pattern_b, pattern_c_macd, pattern_e_boll, pattern_f_vwap, pattern_g_vcp
from stock_triggers.ui.patterns.penalties import apply_signal_penalty_weights, compute_signal_penalty_features, get_recent_signal_lookback_days, load_signal_penalty_weights
from stock_triggers.ui.patterns.scoring import apply_pattern_family_bonus, clip_score

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_SUMMARY_OUT = DATA_DIR / "combo_analysis_summary.csv"
DEFAULT_EVENTS_OUT = DATA_DIR / "combo_analysis_events.csv"
DEFAULT_RAW_OUT = DATA_DIR / "combo_analysis_raw_signals.csv"
TARGET_PAIR_FAMILIES = {
    "A+C": ("A", "C"),
    "B+F": ("B", "F"),
    "E+G": ("E", "G"),
}
TARGET_FAMILIES = tuple(sorted({family for families in TARGET_PAIR_FAMILIES.values() for family in families}))


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Prices file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in prices file: {missing}")
    df.sort_values(["Ticker", "Date"], inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact pattern combos versus single-pattern cohorts")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--summary-out", type=str, default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--events-out", type=str, default=str(DEFAULT_EVENTS_OUT))
    parser.add_argument("--raw-out", type=str, default="")
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--max-hold-days", type=int, default=30)
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)
    parser.add_argument("--pullback-buffer-pct", type=float, default=1.5)
    parser.add_argument("--rebound-min-pct", type=float, default=0.2)
    parser.add_argument("--consensus-bonus", type=float, default=5.0)
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--write-raw-history", action="store_true")
    return parser.parse_args()


def _detect_for_family(
    prices: pd.DataFrame,
    *,
    family: str,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float,
    rebound_min_pct: float,
) -> pd.DataFrame:
    if family == "A":
        detected = pattern_a.detect(
            prices,
            as_of_date=as_of_date,
            breakout_days=int(breakout_days),
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
        )
        return _score_pattern_a_rows(
            detected,
            prices,
            as_of_date=as_of_date,
            breakout_days=int(breakout_days),
        )
    if family == "B":
        return pattern_b.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            pullback_buffer_pct=float(pullback_buffer_pct),
            rebound_min_pct=float(rebound_min_pct),
            compute_rsi_fn=compute_rsi,
        )
    if family == "C":
        return pattern_c_macd.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
    if family == "E":
        return pattern_e_boll.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
    if family == "F":
        return pattern_f_vwap.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=float(volume_multiplier),
            stop_pct=float(stop_pct),
            compute_rsi_fn=compute_rsi,
        )
    if family == "G":
        return pattern_g_vcp.detect(
            prices,
            as_of_date=as_of_date,
            volume_multiplier=min(float(volume_multiplier), 1.2),
            stop_pct=float(stop_pct),
            base_lookback=100,
            dryup_volume_ratio=1.0,
            compute_rsi_fn=compute_rsi,
        )
    return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)


def build_raw_signal_history(
    prices: pd.DataFrame,
    *,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
    pullback_buffer_pct: float,
    rebound_min_pct: float,
    consensus_bonus: float,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    all_dates = pd.Series(sorted(prices["Date"].dropna().unique()))
    if start_date is not None:
        all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]

    rows: list[pd.DataFrame] = []
    pattern_weights = load_pattern_weights()

    for signal_date in all_dates.tolist():
        day_rows: list[pd.DataFrame] = []
        for family in TARGET_FAMILIES:
            detected = _detect_for_family(
                prices,
                family=family,
                as_of_date=pd.Timestamp(signal_date),
                breakout_days=int(breakout_days),
                volume_multiplier=float(volume_multiplier),
                stop_pct=float(stop_pct),
                pullback_buffer_pct=float(pullback_buffer_pct),
                rebound_min_pct=float(rebound_min_pct),
            )
            if not detected.empty:
                day_rows.append(detected)

        if not day_rows:
            continue

        day = pd.concat(day_rows, ignore_index=True)
        day["consensus_count"] = day.groupby(["signal_date", "ticker"])["pattern_family"].transform("nunique")
        if float(consensus_bonus) > 0:
            mask = day["consensus_count"] > 1
            day.loc[mask, "signal_score"] = (
                pd.to_numeric(day.loc[mask, "signal_score"], errors="coerce").fillna(0.0)
                + float(consensus_bonus)
            ).map(clip_score)
        day = apply_pattern_family_bonus(day, pattern_weights)
        rows.append(day.reindex(columns=STANDARD_SIGNAL_COLS).copy())

    if not rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)

    history = pd.concat(rows, ignore_index=True)
    penalty_payload = load_signal_penalty_weights()
    history = compute_signal_penalty_features(
        history,
        prices,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=get_recent_signal_lookback_days(penalty_payload),
    )
    history = apply_signal_penalty_weights(history, penalty_payload)
    history.sort_values(["signal_date", "ticker", "pattern_family"], inplace=True)
    return history.reset_index(drop=True)


def _build_combo_metadata(raw_signals: pd.DataFrame) -> pd.DataFrame:
    if raw_signals.empty:
        return pd.DataFrame(
            columns=["signal_date", "ticker", "families", "family_count", "combo_key", "is_target_combo"]
        )

    grouped = (
        raw_signals.groupby(["signal_date", "ticker"], sort=True)["pattern_family"]
        .apply(lambda series: tuple(sorted({str(value).strip().upper() for value in series if str(value).strip()})))
        .reset_index(name="families")
    )
    grouped["family_count"] = grouped["families"].map(len)
    grouped["combo_key"] = grouped["families"].map(lambda families: "+".join(families))
    grouped["is_target_combo"] = grouped["combo_key"].isin(TARGET_PAIR_FAMILIES)
    return grouped


def _cohort_label(combo_key: str, family_count: int, representative_family: str) -> str | None:
    clean_family = str(representative_family).strip().upper()
    if combo_key in TARGET_PAIR_FAMILIES:
        return combo_key
    if int(family_count) == 1 and clean_family in TARGET_FAMILIES:
        return f"{clean_family}_only"
    return None


def build_event_view(raw_signals: pd.DataFrame) -> pd.DataFrame:
    if raw_signals.empty:
        return pd.DataFrame()

    combo_meta = _build_combo_metadata(raw_signals)
    ranked = raw_signals.copy()
    ranked["signal_score"] = pd.to_numeric(ranked["signal_score"], errors="coerce")
    ranked.sort_values(
        ["signal_date", "ticker", "signal_score", "pattern_family"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    top_rows = ranked.drop_duplicates(subset=["signal_date", "ticker"], keep="first").copy()
    events = top_rows.merge(combo_meta, on=["signal_date", "ticker"], how="left")
    events["cohort"] = events.apply(
        lambda row: _cohort_label(
            combo_key=str(row.get("combo_key", "")),
            family_count=int(row.get("family_count", 0) or 0),
            representative_family=str(row.get("pattern_family", "")),
        ),
        axis=1,
    )
    events = events[events["cohort"].notna()].copy()
    events["month"] = pd.to_datetime(events["signal_date"], errors="coerce").dt.to_period("M").astype("string")
    return events.reset_index(drop=True)


def simulate_trade_outcomes(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices.groupby("Ticker", sort=False)}

    statuses: list[str] = []
    exit_dates: list[str | None] = []
    exit_prices: list[float | None] = []
    days_held_list: list[int] = []
    return_pct_list: list[float | None] = []

    for _, row in events_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
        stop_price = pd.to_numeric(row.get("stop_price"), errors="coerce")
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"
        hist = grouped.get(ticker)
        if hist is None and ticker.endswith(".NS"):
            hist = grouped.get(ticker[:-3])

        status = "hold"
        exit_date = None
        exit_price = None
        days_held = 0
        return_pct = None

        if hist is not None and pd.notna(signal_date) and pd.notna(entry_price) and float(entry_price) > 0:
            if pd.isna(stop_price) or float(stop_price) <= 0 or float(stop_price) >= float(entry_price):
                stop_price = float(entry_price) * (1.0 - float(stop_pct) / 100.0)
            target_price = float(entry_price) * (1.0 + float(target_pct) / 100.0)
            future = hist[hist["Date"] > signal_date].head(int(max_hold_days)).copy()
            for bar_number, (_, bar) in enumerate(future.iterrows(), start=1):
                low_value = pd.to_numeric(bar.get("Low"), errors="coerce")
                high_value = pd.to_numeric(bar.get("High"), errors="coerce")
                close_value = pd.to_numeric(bar.get("Close"), errors="coerce")
                bar_date = pd.to_datetime(bar.get("Date"), errors="coerce")
                if pd.notna(low_value) and float(low_value) <= float(stop_price):
                    status = "stop"
                    exit_date = bar_date.date().isoformat() if pd.notna(bar_date) else None
                    exit_price = float(stop_price)
                    days_held = bar_number
                    break
                if pd.notna(high_value) and float(high_value) >= target_price:
                    status = "target"
                    exit_date = bar_date.date().isoformat() if pd.notna(bar_date) else None
                    exit_price = float(target_price)
                    days_held = bar_number
                    break
                if bar_number == len(future) and pd.notna(close_value):
                    status = "hold"
                    exit_date = bar_date.date().isoformat() if pd.notna(bar_date) else None
                    exit_price = float(close_value)
                    days_held = bar_number

            if exit_price is not None:
                return_pct = ((float(exit_price) / float(entry_price)) - 1.0) * 100.0

        statuses.append(status)
        exit_dates.append(exit_date)
        exit_prices.append(round(float(exit_price), 4) if exit_price is not None else None)
        days_held_list.append(int(days_held))
        return_pct_list.append(round(float(return_pct), 4) if return_pct is not None else None)

    out = events_df.copy()
    out["trade_status"] = statuses
    out["trade_exit_date"] = exit_dates
    out["trade_exit_price"] = exit_prices
    out["trade_days_held"] = days_held_list
    out["trade_return_pct"] = return_pct_list
    return out


def summarize_cohorts(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for cohort, group in events_df.groupby("cohort", sort=True):
        target_count = int((group["trade_status"] == "target").sum())
        stop_count = int((group["trade_status"] == "stop").sum())
        hold_count = int((group["trade_status"] == "hold").sum())
        closed_count = target_count + stop_count
        rows.append(
            {
                "cohort": cohort,
                "events": int(len(group)),
                "months": int(group["month"].nunique()),
                "target_hits": target_count,
                "stop_hits": stop_count,
                "holds": hold_count,
                "win_rate_closed_pct": round((target_count / closed_count) * 100.0, 2) if closed_count else np.nan,
                "target_hit_rate_pct": round((target_count / len(group)) * 100.0, 2),
                "stop_hit_rate_pct": round((stop_count / len(group)) * 100.0, 2),
                "avg_return_pct": round(pd.to_numeric(group["trade_return_pct"], errors="coerce").mean(), 4),
                "median_return_pct": round(pd.to_numeric(group["trade_return_pct"], errors="coerce").median(), 4),
                "avg_days_held": round(pd.to_numeric(group["trade_days_held"], errors="coerce").mean(), 2),
                "median_days_held": round(pd.to_numeric(group["trade_days_held"], errors="coerce").median(), 2),
                "avg_signal_score": round(pd.to_numeric(group["signal_score"], errors="coerce").mean(), 2),
                "median_signal_score": round(pd.to_numeric(group["signal_score"], errors="coerce").median(), 2),
            }
        )

    summary = pd.DataFrame(rows)
    desired_order = ["A+C", "A_only", "C_only", "B+F", "B_only", "F_only", "E+G", "E_only", "G_only"]
    summary["sort_key"] = summary["cohort"].map({name: idx for idx, name in enumerate(desired_order)}).fillna(999).astype(int)
    summary.sort_values(["sort_key", "cohort"], inplace=True)
    return summary.drop(columns=["sort_key"]).reset_index(drop=True)


def build_pair_comparisons(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    summary_lookup = summary_df.set_index("cohort")
    for pair_name, families in TARGET_PAIR_FAMILIES.items():
        if pair_name not in summary_lookup.index:
            continue
        pair_row = summary_lookup.loc[pair_name]
        single_names = [f"{family}_only" for family in families]
        singles = summary_lookup.loc[[name for name in single_names if name in summary_lookup.index]].copy()
        if singles.empty:
            continue
        avg_single_return = pd.to_numeric(singles["avg_return_pct"], errors="coerce").mean()
        avg_single_win = pd.to_numeric(singles["win_rate_closed_pct"], errors="coerce").mean()
        best_single_return = pd.to_numeric(singles["avg_return_pct"], errors="coerce").max()
        best_single_win = pd.to_numeric(singles["win_rate_closed_pct"], errors="coerce").max()
        rows.append(
            {
                "pair": pair_name,
                "pair_events": int(pair_row["events"]),
                "pair_avg_return_pct": round(float(pair_row["avg_return_pct"]), 4) if pd.notna(pair_row["avg_return_pct"]) else np.nan,
                "pair_win_rate_closed_pct": round(float(pair_row["win_rate_closed_pct"]), 2) if pd.notna(pair_row["win_rate_closed_pct"]) else np.nan,
                "avg_single_return_pct": round(float(avg_single_return), 4) if pd.notna(avg_single_return) else np.nan,
                "best_single_return_pct": round(float(best_single_return), 4) if pd.notna(best_single_return) else np.nan,
                "avg_single_win_rate_closed_pct": round(float(avg_single_win), 2) if pd.notna(avg_single_win) else np.nan,
                "best_single_win_rate_closed_pct": round(float(best_single_win), 2) if pd.notna(best_single_win) else np.nan,
                "pair_minus_avg_single_return_pct": round(float(pair_row["avg_return_pct"] - avg_single_return), 4) if pd.notna(pair_row["avg_return_pct"]) and pd.notna(avg_single_return) else np.nan,
                "pair_minus_best_single_return_pct": round(float(pair_row["avg_return_pct"] - best_single_return), 4) if pd.notna(pair_row["avg_return_pct"]) and pd.notna(best_single_return) else np.nan,
                "pair_minus_avg_single_win_rate_pct": round(float(pair_row["win_rate_closed_pct"] - avg_single_win), 2) if pd.notna(pair_row["win_rate_closed_pct"]) and pd.notna(avg_single_win) else np.nan,
                "pair_minus_best_single_win_rate_pct": round(float(pair_row["win_rate_closed_pct"] - best_single_win), 2) if pd.notna(pair_row["win_rate_closed_pct"]) and pd.notna(best_single_win) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    prices = load_prices(Path(args.prices))
    start_date = pd.to_datetime(args.start_date, errors="coerce") if args.start_date else None
    end_date = pd.to_datetime(args.end_date, errors="coerce") if args.end_date else None
    raw_signals = build_raw_signal_history(
        prices,
        breakout_days=int(args.breakout_days),
        volume_multiplier=float(args.volume_multiplier),
        stop_pct=float(args.stop_pct),
        pullback_buffer_pct=float(args.pullback_buffer_pct),
        rebound_min_pct=float(args.rebound_min_pct),
        consensus_bonus=float(args.consensus_bonus),
        start_date=start_date,
        end_date=end_date,
    )
    events = build_event_view(raw_signals)
    labels = compute_stop_event_labels(
        events,
        prices,
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        max_hold_days=int(args.max_hold_days),
        require_full_horizon=True,
    )
    if not labels.empty:
        labels["signal_date"] = pd.to_datetime(labels["signal_date"], errors="coerce").dt.date.astype("string")
        events = events.merge(
            labels,
            on=["ticker", "signal_date", "pattern_family"],
            how="left",
        )
    events = simulate_trade_outcomes(
        events,
        prices,
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        max_hold_days=int(args.max_hold_days),
    )
    summary = summarize_cohorts(events)
    comparisons = build_pair_comparisons(summary)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_path = summary_out.with_name(summary_out.stem + "_pairs" + summary_out.suffix)
    events_out = Path(args.events_out)
    events_out.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_out, index=False)
    comparisons.to_csv(comparison_path, index=False)
    events.to_csv(events_out, index=False)

    if args.write_raw_history:
        raw_out = Path(args.raw_out) if args.raw_out else Path(DEFAULT_RAW_OUT)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_signals.to_csv(raw_out, index=False)

    payload = {
        "summary_out": str(summary_out),
        "pair_comparison_out": str(comparison_path),
        "events_out": str(events_out),
        "raw_signal_rows": int(len(raw_signals)),
        "event_rows": int(len(events)),
        "cohorts": summary.to_dict(orient="records"),
        "pair_comparisons": comparisons.to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()