"""Leakage-safe monthly walk-forward evaluation for stop-risk feature sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.compute_signal_stop_risk_model import compute_stop_event_labels, compute_stop_risk_model
from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, parse_optional_date
from stock_triggers.ui.patterns.stop_risk import STOP_RISK_FEATURE_SET_PRESETS, apply_signal_stop_risk_model

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"

CANDIDATE_SPECS = {
    "full": {"feature_set": "full", "include_family_features": True},
    "full_no_family": {"feature_set": "full", "include_family_features": False},
    "scores_only": {"feature_set": "scores_only", "include_family_features": True},
    "scores_only_no_family": {"feature_set": "scores_only", "include_family_features": False},
    "scores_plus_row_context": {"feature_set": "scores_plus_row_context", "include_family_features": True},
    "scores_plus_regime": {"feature_set": "scores_plus_regime", "include_family_features": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stop-risk feature sets with monthly walk-forward evaluation")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--max-hold-days", type=int, default=30)
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument("--recent-signal-lookback-days", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--tail-quantile", type=float, default=0.2)
    parser.add_argument("--train-start-date", type=str, default="", help="Only use signals on or after this date (YYYY-MM-DD)")
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        choices=["walk-forward", "holdout"],
        default="walk-forward",
        help="Use monthly expanding walk-forward or a single train-on-past/test-on-rest holdout split",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="",
        help="Inclusive train cutoff for holdout mode (YYYY-MM-DD). Test rows come after this date.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=sorted(CANDIDATE_SPECS),
        help="Candidate spec to evaluate. Repeat to compare multiple. Defaults to a standard set.",
    )
    parser.add_argument("--summary-out", type=str, default="")
    parser.add_argument("--monthly-out", type=str, default="")
    parser.add_argument("--predictions-out", type=str, default="")
    parser.add_argument("--recency-half-life-months", type=float, default=0.0, help="Half-life in months for recency weighting during training. 0 disables weighting.")
    return parser.parse_args()


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return float("nan")
    left_rank = frame["left"].rank(method="average")
    right_rank = frame["right"].rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def _resolve_candidates(raw_candidates: list[str] | None) -> list[str]:
    if raw_candidates:
        return list(dict.fromkeys(raw_candidates))
    return [
        "full",
        "full_no_family",
        "scores_only",
        "scores_plus_row_context",
        "scores_plus_regime",
    ]


def _load_signals(signals_path: Path, prices_df: pd.DataFrame, *, max_hold_days: int) -> pd.DataFrame:
    signals = pd.read_csv(signals_path)
    signals["signal_date"] = pd.to_datetime(signals.get("signal_date"), errors="coerce")
    signals = signals.dropna(subset=["signal_date", "ticker", "pattern_family", "entry_price"]).copy()
    latest_price_date = pd.to_datetime(prices_df["Date"], errors="coerce").max()
    complete_cutoff = pd.Timestamp(latest_price_date) - BDay(int(max_hold_days))
    signals = signals.loc[signals["signal_date"] <= complete_cutoff].copy()
    signals["month"] = signals["signal_date"].dt.to_period("M").astype(str)
    signals.sort_values(["signal_date", "ticker", "pattern_family"], inplace=True)
    return signals.reset_index(drop=True)


def _compute_labels(signals_df: pd.DataFrame, prices_df: pd.DataFrame, *, target_pct: float, stop_pct: float, max_hold_days: int) -> pd.DataFrame:
    labels = compute_stop_event_labels(
        signals_df,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        require_full_horizon=True,
    )
    labels["signal_date"] = pd.to_datetime(labels["signal_date"], errors="coerce")
    labels["month"] = labels["signal_date"].dt.to_period("M").astype(str)
    return labels


def _baseline_metrics(predictions_df: pd.DataFrame, *, tail_quantile: float) -> dict:
    baseline = predictions_df.copy()
    baseline["signal_score"] = pd.to_numeric(baseline["signal_score"], errors="coerce")
    baseline["stop_before_target"] = pd.to_numeric(baseline["stop_before_target"], errors="coerce")
    baseline = baseline.dropna(subset=["signal_score", "stop_before_target"])
    if baseline.empty:
        return {}

    bottom_cut = float(baseline["signal_score"].quantile(float(tail_quantile)))
    top_cut = float(baseline["signal_score"].quantile(1.0 - float(tail_quantile)))
    bottom = baseline.loc[baseline["signal_score"] <= bottom_cut]
    top = baseline.loc[baseline["signal_score"] >= top_cut]
    bottom_rate = float(bottom["stop_before_target"].mean()) if not bottom.empty else np.nan
    top_rate = float(top["stop_before_target"].mean()) if not top.empty else np.nan
    return {
        "baseline_spearman_signal_score_vs_stop": round(_spearman_corr(baseline["signal_score"], baseline["stop_before_target"]), 4),
        "baseline_top_cut": round(top_cut, 4),
        "baseline_bottom_cut": round(bottom_cut, 4),
        "baseline_top_stop_rate": round(top_rate, 4) if not np.isnan(top_rate) else np.nan,
        "baseline_bottom_stop_rate": round(bottom_rate, 4) if not np.isnan(bottom_rate) else np.nan,
    }


def evaluate_candidate(
    candidate_name: str,
    spec: dict,
    signals_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    breakout_days: int,
    recent_signal_lookback_days: int,
    min_train_rows: int,
    tail_quantile: float,
    evaluation_mode: str,
    train_end_date: pd.Timestamp | None,
    recency_half_life_months: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if str(evaluation_mode) == "holdout":
        return evaluate_candidate_holdout(
            candidate_name,
            spec,
            signals_df,
            labels_df,
            prices_df,
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            breakout_days=int(breakout_days),
            recent_signal_lookback_days=int(recent_signal_lookback_days),
            min_train_rows=int(min_train_rows),
            tail_quantile=float(tail_quantile),
            train_end_date=train_end_date,
            recency_half_life_months=float(recency_half_life_months),
        )

    months = sorted(month for month in signals_df["month"].dropna().unique().tolist() if month)
    merged_months: list[pd.DataFrame] = []
    monthly_rows: list[dict] = []

    for month in months:
        test_rows = signals_df.loc[signals_df["month"] == month].copy()
        if test_rows.empty:
            continue
        month_start = test_rows["signal_date"].min()
        train_rows = signals_df.loc[signals_df["signal_date"] < month_start].copy()
        if len(train_rows) < int(min_train_rows):
            continue
        if float(recency_half_life_months) > 0:
            train_rows = add_recency_weights(
                train_rows,
                date_col="signal_date",
                half_life_months=float(recency_half_life_months),
                reference_date=month_start,
            )

        payload = compute_stop_risk_model(
            train_rows,
            prices_df,
            breakout_days=int(breakout_days),
            recent_signal_lookback_days=int(recent_signal_lookback_days),
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            numeric_features=list(STOP_RISK_FEATURE_SET_PRESETS[spec["feature_set"]]),
            include_family_features=bool(spec["include_family_features"]),
            feature_set_name=str(candidate_name),
            require_full_horizon=True,
        )
        if int(payload.get("signals_analyzed", 0)) == 0:
            continue

        context_rows = signals_df.loc[signals_df["signal_date"] <= test_rows["signal_date"].max()].copy()
        scored_context = apply_signal_stop_risk_model(
            context_rows,
            prices_df,
            payload,
            breakout_days=int(breakout_days),
        )
        scored_context["signal_date"] = pd.to_datetime(scored_context["signal_date"], errors="coerce")
        scored_context["month"] = scored_context["signal_date"].dt.to_period("M").astype(str)
        scored_test = scored_context.loc[scored_context["month"] == month].copy()

        month_labels = labels_df.loc[
            labels_df["month"] == month,
            ["ticker", "signal_date", "pattern_family", "stop_before_target"],
        ].copy()
        merged = scored_test.merge(month_labels, on=["ticker", "signal_date", "pattern_family"], how="inner")
        if merged.empty:
            continue

        merged["candidate_name"] = candidate_name
        merged_months.append(merged)

        month_risk = pd.to_numeric(merged["signal_stop_risk"], errors="coerce")
        month_stop = pd.to_numeric(merged["stop_before_target"], errors="coerce")
        low_cut = float(month_risk.quantile(float(tail_quantile)))
        high_cut = float(month_risk.quantile(1.0 - float(tail_quantile)))
        low_risk = merged.loc[month_risk <= low_cut]
        high_risk = merged.loc[month_risk >= high_cut]
        monthly_rows.append(
            {
                "candidate_name": candidate_name,
                "month": month,
                "train_rows": int(payload.get("signals_analyzed", 0)),
                "test_rows": int(len(merged)),
                "spearman_stop_risk_vs_stop": round(_spearman_corr(month_risk, month_stop), 4),
                "avg_signal_stop_risk": round(float(month_risk.mean()), 4),
                "realized_stop_rate": round(float(month_stop.mean()), 4),
                "low20_stop_risk_cutoff": round(low_cut, 4),
                "high20_stop_risk_cutoff": round(high_cut, 4),
                "low20_stop_rate": round(float(pd.to_numeric(low_risk["stop_before_target"], errors="coerce").mean()), 4) if not low_risk.empty else np.nan,
                "high20_stop_rate": round(float(pd.to_numeric(high_risk["stop_before_target"], errors="coerce").mean()), 4) if not high_risk.empty else np.nan,
            }
        )

    if not merged_months:
        empty_summary = {
            "candidate_name": candidate_name,
            "feature_set": spec["feature_set"],
            "include_family_features": bool(spec["include_family_features"]),
            "oos_rows": 0,
            "months": 0,
        }
        return empty_summary, pd.DataFrame(monthly_rows), pd.DataFrame()

    predictions = pd.concat(merged_months, ignore_index=True)
    predictions["signal_stop_risk"] = pd.to_numeric(predictions["signal_stop_risk"], errors="coerce")
    predictions["stop_before_target"] = pd.to_numeric(predictions["stop_before_target"], errors="coerce")
    predictions["signal_score"] = pd.to_numeric(predictions["signal_score"], errors="coerce")
    predictions = predictions.dropna(subset=["signal_stop_risk", "stop_before_target", "signal_score"]).copy()

    low_cut = float(predictions["signal_stop_risk"].quantile(float(tail_quantile)))
    high_cut = float(predictions["signal_stop_risk"].quantile(1.0 - float(tail_quantile)))
    low_risk = predictions.loc[predictions["signal_stop_risk"] <= low_cut]
    high_risk = predictions.loc[predictions["signal_stop_risk"] >= high_cut]
    low_rate = float(low_risk["stop_before_target"].mean()) if not low_risk.empty else np.nan
    high_rate = float(high_risk["stop_before_target"].mean()) if not high_risk.empty else np.nan
    baseline = _baseline_metrics(predictions, tail_quantile=float(tail_quantile))

    summary = {
        "candidate_name": candidate_name,
        "feature_set": spec["feature_set"],
        "include_family_features": bool(spec["include_family_features"]),
        "oos_rows": int(len(predictions)),
        "months": int(predictions["month"].nunique()),
        "spearman_stop_risk_vs_stop": round(_spearman_corr(predictions["signal_stop_risk"], predictions["stop_before_target"]), 4),
        "low20_stop_risk_cutoff": round(low_cut, 4),
        "high20_stop_risk_cutoff": round(high_cut, 4),
        "low20_stop_rate": round(low_rate, 4),
        "high20_stop_rate": round(high_rate, 4),
        "stop_rate_gap_high20_minus_low20": round(float(high_rate - low_rate), 4),
        "relative_lift_high_vs_low": round(float((high_rate / low_rate) - 1.0), 4) if low_rate not in (0.0, np.nan) and not np.isnan(low_rate) else np.nan,
    }
    summary.update(baseline)
    return summary, pd.DataFrame(monthly_rows), predictions


def evaluate_candidate_holdout(
    candidate_name: str,
    spec: dict,
    signals_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    breakout_days: int,
    recent_signal_lookback_days: int,
    min_train_rows: int,
    tail_quantile: float,
    train_end_date: pd.Timestamp | None,
    recency_half_life_months: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if train_end_date is None:
        raise SystemExit("--train-end-date is required when --evaluation-mode=holdout")

    train_rows = signals_df.loc[signals_df["signal_date"] <= train_end_date].copy()
    test_rows = signals_df.loc[signals_df["signal_date"] > train_end_date].copy()
    if len(train_rows) < int(min_train_rows) or test_rows.empty:
        summary = {
            "candidate_name": candidate_name,
            "feature_set": spec["feature_set"],
            "include_family_features": bool(spec["include_family_features"]),
            "oos_rows": 0,
            "months": 0,
            "evaluation_mode": "holdout",
            "train_end_date": train_end_date.date().isoformat(),
        }
        return summary, pd.DataFrame(), pd.DataFrame()
    if float(recency_half_life_months) > 0:
        train_rows = add_recency_weights(
            train_rows,
            date_col="signal_date",
            half_life_months=float(recency_half_life_months),
            reference_date=train_end_date,
        )

    payload = compute_stop_risk_model(
        train_rows,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(recent_signal_lookback_days),
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        numeric_features=list(STOP_RISK_FEATURE_SET_PRESETS[spec["feature_set"]]),
        include_family_features=bool(spec["include_family_features"]),
        feature_set_name=str(candidate_name),
        require_full_horizon=True,
    )
    if int(payload.get("signals_analyzed", 0)) == 0:
        summary = {
            "candidate_name": candidate_name,
            "feature_set": spec["feature_set"],
            "include_family_features": bool(spec["include_family_features"]),
            "oos_rows": 0,
            "months": 0,
            "evaluation_mode": "holdout",
            "train_end_date": train_end_date.date().isoformat(),
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    context_rows = signals_df.copy()
    scored_context = apply_signal_stop_risk_model(
        context_rows,
        prices_df,
        payload,
        breakout_days=int(breakout_days),
    )
    scored_context["signal_date"] = pd.to_datetime(scored_context["signal_date"], errors="coerce")
    scored_context["month"] = scored_context["signal_date"].dt.to_period("M").astype(str)
    scored_test = scored_context.loc[scored_context["signal_date"] > train_end_date].copy()

    test_labels = labels_df.loc[
        labels_df["signal_date"] > train_end_date,
        ["ticker", "signal_date", "pattern_family", "stop_before_target", "month"],
    ].copy()
    predictions = scored_test.merge(test_labels, on=["ticker", "signal_date", "pattern_family", "month"], how="inner")
    if predictions.empty:
        summary = {
            "candidate_name": candidate_name,
            "feature_set": spec["feature_set"],
            "include_family_features": bool(spec["include_family_features"]),
            "oos_rows": 0,
            "months": 0,
            "evaluation_mode": "holdout",
            "train_end_date": train_end_date.date().isoformat(),
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    predictions["candidate_name"] = candidate_name
    predictions["signal_stop_risk"] = pd.to_numeric(predictions["signal_stop_risk"], errors="coerce")
    predictions["stop_before_target"] = pd.to_numeric(predictions["stop_before_target"], errors="coerce")
    predictions["signal_score"] = pd.to_numeric(predictions["signal_score"], errors="coerce")
    predictions = predictions.dropna(subset=["signal_stop_risk", "stop_before_target", "signal_score"]).copy()

    low_cut = float(predictions["signal_stop_risk"].quantile(float(tail_quantile)))
    high_cut = float(predictions["signal_stop_risk"].quantile(1.0 - float(tail_quantile)))
    low_risk = predictions.loc[predictions["signal_stop_risk"] <= low_cut]
    high_risk = predictions.loc[predictions["signal_stop_risk"] >= high_cut]
    low_rate = float(low_risk["stop_before_target"].mean()) if not low_risk.empty else np.nan
    high_rate = float(high_risk["stop_before_target"].mean()) if not high_risk.empty else np.nan
    baseline = _baseline_metrics(predictions, tail_quantile=float(tail_quantile))

    monthly_df = (
        predictions.groupby("month", sort=True)
        .agg(
            train_rows=("stop_before_target", lambda _: int(payload.get("signals_analyzed", 0))),
            test_rows=("stop_before_target", "size"),
            spearman_stop_risk_vs_stop=("signal_stop_risk", lambda col: round(_spearman_corr(col, predictions.loc[col.index, "stop_before_target"]), 4)),
            avg_signal_stop_risk=("signal_stop_risk", "mean"),
            realized_stop_rate=("stop_before_target", "mean"),
        )
        .reset_index()
    )
    monthly_df.insert(0, "candidate_name", candidate_name)
    monthly_df["train_end_date"] = train_end_date.date().isoformat()
    monthly_df["evaluation_mode"] = "holdout"
    monthly_df["avg_signal_stop_risk"] = monthly_df["avg_signal_stop_risk"].round(4)
    monthly_df["realized_stop_rate"] = monthly_df["realized_stop_rate"].round(4)

    summary = {
        "candidate_name": candidate_name,
        "feature_set": spec["feature_set"],
        "include_family_features": bool(spec["include_family_features"]),
        "oos_rows": int(len(predictions)),
        "months": int(predictions["month"].nunique()),
        "evaluation_mode": "holdout",
        "train_end_date": train_end_date.date().isoformat(),
        "train_rows": int(payload.get("signals_analyzed", 0)),
        "test_rows": int(len(predictions)),
        "spearman_stop_risk_vs_stop": round(_spearman_corr(predictions["signal_stop_risk"], predictions["stop_before_target"]), 4),
        "low20_stop_risk_cutoff": round(low_cut, 4),
        "high20_stop_risk_cutoff": round(high_cut, 4),
        "low20_stop_rate": round(low_rate, 4),
        "high20_stop_rate": round(high_rate, 4),
        "stop_rate_gap_high20_minus_low20": round(float(high_rate - low_rate), 4),
        "relative_lift_high_vs_low": round(float((high_rate / low_rate) - 1.0), 4) if low_rate not in (0.0, np.nan) and not np.isnan(low_rate) else np.nan,
    }
    summary.update(baseline)
    return summary, monthly_df, predictions


def main() -> None:
    args = parse_args()
    prices_path = Path(args.prices)
    signals_path = Path(args.signals)
    if not prices_path.exists():
        raise SystemExit(f"Prices file not found: {prices_path}")
    if not signals_path.exists():
        raise SystemExit(f"Signals file not found: {signals_path}")

    prices = pd.read_csv(prices_path, parse_dates=["Date"])
    signals = _load_signals(signals_path, prices, max_hold_days=int(args.max_hold_days))
    train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
    train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
    signals = filter_by_date_window(signals, date_col="signal_date", start_date=train_start_date, end_date=None)
    labels = _compute_labels(
        signals,
        prices,
        target_pct=float(args.target_pct),
        stop_pct=float(args.stop_pct),
        max_hold_days=int(args.max_hold_days),
    )
    candidates = _resolve_candidates(args.candidate)

    summaries: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_name in candidates:
        summary, monthly_df, predictions_df = evaluate_candidate(
            candidate_name,
            CANDIDATE_SPECS[candidate_name],
            signals,
            labels,
            prices,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            max_hold_days=int(args.max_hold_days),
            breakout_days=int(args.breakout_days),
            recent_signal_lookback_days=int(args.recent_signal_lookback_days),
            min_train_rows=int(args.min_train_rows),
            tail_quantile=float(args.tail_quantile),
            evaluation_mode=str(args.evaluation_mode),
            train_end_date=train_end_date,
            recency_half_life_months=float(args.recency_half_life_months),
        )
        summaries.append(summary)
        if not monthly_df.empty:
            monthly_frames.append(monthly_df)
        if not predictions_df.empty:
            prediction_frames.append(predictions_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["spearman_stop_risk_vs_stop", "stop_rate_gap_high20_minus_low20"],
        ascending=[False, False],
        na_position="last",
    )
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    payload = {
        "prices": str(prices_path),
        "signals": str(signals_path),
        "evaluation_mode": str(args.evaluation_mode),
        "train_start_date": train_start_date.date().isoformat() if train_start_date is not None else None,
        "train_end_date": train_end_date.date().isoformat() if train_end_date is not None else None,
        "recency_half_life_months": float(args.recency_half_life_months),
        "candidates": candidates,
        "summary": summary_df.to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))

    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_out, index=False)
    if args.monthly_out and not monthly_df.empty:
        Path(args.monthly_out).parent.mkdir(parents=True, exist_ok=True)
        monthly_df.to_csv(args.monthly_out, index=False)
    if args.predictions_out and not predictions_df.empty:
        Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(args.predictions_out, index=False)


if __name__ == "__main__":
    main()