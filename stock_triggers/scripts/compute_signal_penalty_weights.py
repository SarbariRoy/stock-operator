"""Compute data-driven row-level signal penalty weights from historical outcomes."""

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

from stock_triggers.ui.patterns.penalties import apply_signal_penalty_weights, compute_signal_penalty_features

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_all_patterns.csv"
DEFAULT_TRAINING_DATA = DATA_DIR / "training_signals_history.csv"
DEFAULT_OUTPUT = DATA_DIR / "signal_penalty_weights.json"
PATTERN_FAMILIES = ("A", "B", "C", "D", "E", "F", "G")
FEATURE_NAMES = (
    "feature_recent_signal_count",
    "feature_close_vs_prev_high_pct",
    "feature_close_vs_sma50_pct",
    "feature_gap_pct",
    "feature_range_vs_atr",
)
FEATURE_DIRECTIONS = {
    "feature_recent_signal_count": "higher",
    "feature_close_vs_prev_high_pct": "higher",
    "feature_close_vs_sma50_pct": "higher",
    "feature_gap_pct": "higher",
    "feature_range_vs_atr": "higher",
}
RECENT_SIGNAL_LOOKBACK_CANDIDATES = (5, 10, 20, 40)
DEFAULT_EDGE_PENALTY_SCALE = 50.0
LOOKBACK_SELECTION_SCORE_THRESHOLD = 80.0
LOOKBACK_SELECTION_TOP1_WEIGHT = 0.65
LOOKBACK_SELECTION_THRESHOLD_WEIGHT = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute learned row-level signal penalties from historical signal outcomes")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES))
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS))
    parser.add_argument(
        "--training-data",
        type=str,
        default="",
        help="Optional shared training artifact with precomputed features and outcomes",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=7.0)
    parser.add_argument("--max-hold-days", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--confidence-samples", type=int, default=100)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--breakout-days", type=int, default=40)
    parser.add_argument("--edge-penalty-scale", type=float, default=DEFAULT_EDGE_PENALTY_SCALE)
    parser.add_argument("--recent-signal-lookback-days", type=int, default=0, help="0 = auto-select from candidate windows")
    return parser.parse_args()


def _load_training_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _feature_recent_signal_count_column(lookback_days: int) -> str:
    return f"feature_recent_signal_count_{int(lookback_days)}d"


def _featured_from_training_data(training_df: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
    featured = training_df.copy()
    recent_count_column = _feature_recent_signal_count_column(int(lookback_days))
    if recent_count_column in featured.columns:
        featured["feature_recent_signal_count"] = featured[recent_count_column]
    elif "feature_recent_signal_count" not in featured.columns:
        raise SystemExit(
            "Training data missing recent-signal-count columns required for penalty training"
        )
    featured["signal_date"] = pd.to_datetime(featured["signal_date"], errors="coerce").dt.date.astype("string")
    return featured


def _outcomes_from_training_data(training_df: pd.DataFrame) -> pd.DataFrame:
    outcome_column = "outcome_30d" if "outcome_30d" in training_df.columns else "outcome"
    required = {"ticker", "signal_date", "pattern_family", outcome_column}
    missing = sorted(required - set(training_df.columns))
    if missing:
        raise SystemExit(f"Training data missing required outcome columns: {missing}")
    outcomes = training_df[["ticker", "signal_date", "pattern_family", outcome_column]].copy()
    outcomes.rename(columns={outcome_column: "outcome"}, inplace=True)
    outcomes["signal_date"] = pd.to_datetime(outcomes["signal_date"], errors="coerce").dt.date.astype("string")
    return outcomes


def _resolve_price_history(grouped: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in grouped:
        return grouped[clean]
    if clean.endswith(".NS"):
        return grouped.get(clean[:-3])
    return grouped.get(clean + ".NS")


def _classify_outcomes(
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
        family = str(sig.get("pattern_family", "")).strip().upper()
        signal_date = pd.to_datetime(sig.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(sig.get("entry_price"), errors="coerce")
        if not ticker or family not in PATTERN_FAMILIES or pd.isna(signal_date) or pd.isna(entry_price) or float(entry_price) <= 0:
            continue

        hist = _resolve_price_history(grouped, ticker)
        if hist is None:
            continue
        future = hist[hist["Date"] > signal_date].head(int(max_hold_days))
        if future.empty:
            continue

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
            }
        )

    return pd.DataFrame(rows)


def _evaluate_recent_signal_lookback(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    lookback_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    edge_penalty_scale: float,
) -> tuple[float, dict]:
    featured = compute_signal_penalty_features(
        signals_df,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(lookback_days),
    )
    outcomes = _classify_outcomes(
        featured,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
    )
    if outcomes.empty:
        return float("-inf"), {"lookback_days": int(lookback_days), "signals": 0}

    merged = featured.copy()
    merged["signal_date"] = merged["signal_date"].astype(str)
    merged = merged.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    if merged.empty:
        return float("-inf"), {"lookback_days": int(lookback_days), "signals": int(len(merged))}

    payload = _build_penalty_payload_from_merged(
        merged,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        min_samples=int(min_samples),
        confidence_samples=int(confidence_samples),
        quantiles=int(quantiles),
        recent_signal_lookback_days=int(lookback_days),
        lookback_diagnostics=[],
        edge_penalty_scale=float(edge_penalty_scale),
    )
    scored = apply_signal_penalty_weights(featured.copy(), payload)
    scored["signal_date"] = scored["signal_date"].astype(str)
    scored = scored.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    score_series = pd.to_numeric(scored["signal_score"], errors="coerce")

    threshold_df = scored[score_series >= float(LOOKBACK_SELECTION_SCORE_THRESHOLD)].copy()
    threshold_win_rate = float((threshold_df["outcome"] == "win").mean()) if not threshold_df.empty else 0.0
    threshold_loss_rate = float((threshold_df["outcome"] == "loss").mean()) if not threshold_df.empty else 0.0
    threshold_edge = threshold_win_rate - threshold_loss_rate

    top1_df = scored.copy()
    top1_df["signal_score_numeric"] = score_series
    top1_df.sort_values(["signal_date", "signal_score_numeric", "ticker"], ascending=[True, False, True], inplace=True)
    top1_df = top1_df.drop_duplicates(subset=["signal_date"], keep="first")
    top1_win_rate = float((top1_df["outcome"] == "win").mean()) if not top1_df.empty else 0.0
    top1_loss_rate = float((top1_df["outcome"] == "loss").mean()) if not top1_df.empty else 0.0
    top1_edge = top1_win_rate - top1_loss_rate

    selection_score = (
        float(LOOKBACK_SELECTION_TOP1_WEIGHT) * top1_edge
        + float(LOOKBACK_SELECTION_THRESHOLD_WEIGHT) * threshold_edge
    )

    return selection_score, {
        "lookback_days": int(lookback_days),
        "signals": int(len(scored)),
        "selection_score": round(selection_score, 6),
        "top1_win_rate": round(top1_win_rate, 4),
        "top1_loss_rate": round(top1_loss_rate, 4),
        "top1_edge": round(top1_edge, 4),
        "threshold": int(LOOKBACK_SELECTION_SCORE_THRESHOLD),
        "threshold_count": int(len(threshold_df)),
        "threshold_win_rate": round(threshold_win_rate, 4),
        "threshold_loss_rate": round(threshold_loss_rate, 4),
        "threshold_edge": round(threshold_edge, 4),
    }


def _evaluate_recent_signal_lookback_from_training_data(
    training_df: pd.DataFrame,
    *,
    lookback_days: int,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    edge_penalty_scale: float,
) -> tuple[float, dict]:
    featured = _featured_from_training_data(training_df, lookback_days=int(lookback_days))
    outcomes = _outcomes_from_training_data(training_df)
    merged = featured.copy()
    merged["signal_date"] = merged["signal_date"].astype(str)
    outcomes["signal_date"] = outcomes["signal_date"].astype(str)
    merged = merged.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    if merged.empty:
        return float("-inf"), {"lookback_days": int(lookback_days), "signals": 0}

    payload = _build_penalty_payload_from_merged(
        merged,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        min_samples=int(min_samples),
        confidence_samples=int(confidence_samples),
        quantiles=int(quantiles),
        recent_signal_lookback_days=int(lookback_days),
        lookback_diagnostics=[],
        edge_penalty_scale=float(edge_penalty_scale),
    )
    scored = apply_signal_penalty_weights(featured.copy(), payload)
    scored["signal_date"] = scored["signal_date"].astype(str)
    scored = scored.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    score_series = pd.to_numeric(scored["signal_score"], errors="coerce")

    threshold_df = scored[score_series >= float(LOOKBACK_SELECTION_SCORE_THRESHOLD)].copy()
    threshold_win_rate = float((threshold_df["outcome"] == "win").mean()) if not threshold_df.empty else 0.0
    threshold_loss_rate = float((threshold_df["outcome"] == "loss").mean()) if not threshold_df.empty else 0.0
    threshold_edge = threshold_win_rate - threshold_loss_rate

    top1_df = scored.copy()
    top1_df["signal_score_numeric"] = score_series
    top1_df.sort_values(["signal_date", "signal_score_numeric", "ticker"], ascending=[True, False, True], inplace=True)
    top1_df = top1_df.drop_duplicates(subset=["signal_date"], keep="first")
    top1_win_rate = float((top1_df["outcome"] == "win").mean()) if not top1_df.empty else 0.0
    top1_loss_rate = float((top1_df["outcome"] == "loss").mean()) if not top1_df.empty else 0.0
    top1_edge = top1_win_rate - top1_loss_rate

    selection_score = (
        float(LOOKBACK_SELECTION_TOP1_WEIGHT) * top1_edge
        + float(LOOKBACK_SELECTION_THRESHOLD_WEIGHT) * threshold_edge
    )

    return selection_score, {
        "lookback_days": int(lookback_days),
        "signals": int(len(scored)),
        "selection_score": round(selection_score, 6),
        "top1_win_rate": round(top1_win_rate, 4),
        "top1_loss_rate": round(top1_loss_rate, 4),
        "top1_edge": round(top1_edge, 4),
        "threshold": int(LOOKBACK_SELECTION_SCORE_THRESHOLD),
        "threshold_count": int(len(threshold_df)),
        "threshold_win_rate": round(threshold_win_rate, 4),
        "threshold_loss_rate": round(threshold_loss_rate, 4),
        "threshold_edge": round(threshold_edge, 4),
    }


def _select_recent_signal_lookback_days(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    breakout_days: int,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    edge_penalty_scale: float,
) -> tuple[int, list[dict]]:
    diagnostics: list[dict] = []
    best_window = RECENT_SIGNAL_LOOKBACK_CANDIDATES[0]
    best_score = float("-inf")

    for window in RECENT_SIGNAL_LOOKBACK_CANDIDATES:
        score, detail = _evaluate_recent_signal_lookback(
            signals_df,
            prices_df,
            breakout_days=int(breakout_days),
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            lookback_days=int(window),
            min_samples=int(min_samples),
            confidence_samples=int(confidence_samples),
            quantiles=int(quantiles),
            edge_penalty_scale=float(edge_penalty_scale),
        )
        diagnostics.append(detail)
        if score > best_score:
            best_score = score
            best_window = int(window)

    return best_window, diagnostics


def _select_recent_signal_lookback_days_from_training_data(
    training_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    edge_penalty_scale: float,
) -> tuple[int, list[dict]]:
    diagnostics: list[dict] = []
    best_window = RECENT_SIGNAL_LOOKBACK_CANDIDATES[0]
    best_score = float("-inf")

    for window in RECENT_SIGNAL_LOOKBACK_CANDIDATES:
        score, detail = _evaluate_recent_signal_lookback_from_training_data(
            training_df,
            lookback_days=int(window),
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            min_samples=int(min_samples),
            confidence_samples=int(confidence_samples),
            quantiles=int(quantiles),
            edge_penalty_scale=float(edge_penalty_scale),
        )
        diagnostics.append(detail)
        if score > best_score:
            best_score = score
            best_window = int(window)

    return best_window, diagnostics


def _bucketize(series: pd.Series, *, quantiles: int) -> list[dict]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return []

    quantile_count = max(1, min(int(quantiles), int(values.nunique())))
    if quantile_count <= 1:
        minimum = float(values.min())
        maximum = float(values.max())
        return [{"lower": minimum, "upper": maximum, "include_upper": True}]

    _, bins = pd.qcut(values, q=quantile_count, duplicates="drop", retbins=True)
    if len(bins) < 2:
        minimum = float(values.min())
        maximum = float(values.max())
        return [{"lower": minimum, "upper": maximum, "include_upper": True}]

    buckets: list[dict] = []
    for idx in range(len(bins) - 1):
        buckets.append(
            {
                "lower": float(bins[idx]),
                "upper": float(bins[idx + 1]),
                "include_upper": idx == len(bins) - 2,
            }
        )
    return buckets


def _build_family_feature_mapping(
    df: pd.DataFrame,
    *,
    feature_name: str,
    direction: str,
    quantiles: int,
    min_samples: int,
    confidence_samples: int,
    edge_penalty_scale: float,
) -> dict | None:
    valid = df[pd.to_numeric(df[feature_name], errors="coerce").notna()].copy()
    if len(valid) < int(min_samples):
        return None

    baseline_loss_rate = float((valid["outcome"] == "loss").mean())
    baseline_win_rate = float((valid["outcome"] == "win").mean())
    baseline_edge = baseline_win_rate - baseline_loss_rate
    anchor_value = float(pd.to_numeric(valid[feature_name], errors="coerce").median())
    buckets = _bucketize(valid[feature_name], quantiles=int(quantiles))
    if not buckets:
        return None

    bucket_rows: list[dict] = []
    feature_values = pd.to_numeric(valid[feature_name], errors="coerce")
    for bucket in buckets:
        lower = bucket.get("lower")
        upper = bucket.get("upper")
        include_upper = bool(bucket.get("include_upper", False))
        lower_mask = feature_values >= float(lower)
        if include_upper:
            upper_mask = feature_values <= float(upper)
        else:
            upper_mask = feature_values < float(upper)
        bucket_df = valid[lower_mask & upper_mask].copy()
        if bucket_df.empty:
            continue

        count = len(bucket_df)
        loss_rate = float((bucket_df["outcome"] == "loss").mean())
        win_rate = float((bucket_df["outcome"] == "win").mean())
        bucket_edge = win_rate - loss_rate
        confidence = min(1.0, count / max(1.0, float(confidence_samples)))
        bucket_mid = None
        if lower is not None and upper is not None:
            bucket_mid = (float(lower) + float(upper)) / 2.0
        elif lower is not None:
            bucket_mid = float(lower)
        elif upper is not None:
            bucket_mid = float(upper)

        is_adverse_tail = True
        if direction == "higher":
            is_adverse_tail = bucket_mid is not None and bucket_mid >= anchor_value
        elif direction == "lower":
            is_adverse_tail = bucket_mid is not None and bucket_mid <= anchor_value

        raw_penalty = 0.0
        if is_adverse_tail and bucket_edge < 0.0:
            raw_penalty = max(0.0, (-bucket_edge) * float(edge_penalty_scale) * confidence)
        penalty = -round(raw_penalty, 2)
        bucket_rows.append(
            {
                "lower": None if lower is None else round(float(lower), 6),
                "upper": None if upper is None else round(float(upper), 6),
                "include_upper": include_upper,
                "bucket_mid": None if bucket_mid is None else round(float(bucket_mid), 6),
                "count": count,
                "loss_rate": round(loss_rate, 4),
                "win_rate": round(win_rate, 4),
                "edge": round(bucket_edge, 4),
                "confidence": round(confidence, 4),
                "is_adverse_tail": bool(is_adverse_tail),
                "penalty": penalty,
            }
        )

    if not bucket_rows:
        return None

    return {
        "count": int(len(valid)),
        "baseline_loss_rate": round(baseline_loss_rate, 4),
        "baseline_win_rate": round(baseline_win_rate, 4),
        "baseline_edge": round(baseline_edge, 4),
        "direction": direction,
        "anchor_value": round(anchor_value, 6),
        "buckets": bucket_rows,
    }


def _build_penalty_payload_from_merged(
    merged: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    recent_signal_lookback_days: int,
    lookback_diagnostics: list[dict],
    edge_penalty_scale: float,
) -> dict:
    result = {
        "computed_at": date.today().isoformat(),
        "signals_analyzed": int(len(merged)),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "max_hold_days": int(max_hold_days),
        "quantiles": int(quantiles),
        "edge_penalty_scale": float(edge_penalty_scale),
        "recent_signal_lookback_days": int(recent_signal_lookback_days),
        "recent_signal_lookback_diagnostics": lookback_diagnostics,
        "features": {},
    }

    for feature_name in FEATURE_NAMES:
        feature_spec = {
            "families": {},
            "quantiles": int(quantiles),
        }
        global_mapping = _build_family_feature_mapping(
            merged,
            feature_name=feature_name,
            direction=FEATURE_DIRECTIONS.get(feature_name, "higher"),
            quantiles=int(quantiles),
            min_samples=int(min_samples),
            confidence_samples=int(confidence_samples),
            edge_penalty_scale=float(edge_penalty_scale),
        )
        if global_mapping is not None:
            feature_spec["families"]["__global__"] = global_mapping

        for family in PATTERN_FAMILIES:
            family_mapping = _build_family_feature_mapping(
                merged[merged["pattern_family"] == family].copy(),
                feature_name=feature_name,
                direction=FEATURE_DIRECTIONS.get(feature_name, "higher"),
                quantiles=int(quantiles),
                min_samples=int(min_samples),
                confidence_samples=int(confidence_samples),
                edge_penalty_scale=float(edge_penalty_scale),
            )
            if family_mapping is not None:
                feature_spec["families"][family] = family_mapping

        result["features"][feature_name] = feature_spec

    return result


def compute_penalty_weights(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    breakout_days: int,
    recent_signal_lookback_days: int,
    edge_penalty_scale: float,
) -> dict:
    selected_recent_signal_lookback_days = int(recent_signal_lookback_days)
    lookback_diagnostics: list[dict] = []
    if selected_recent_signal_lookback_days <= 0:
        selected_recent_signal_lookback_days, lookback_diagnostics = _select_recent_signal_lookback_days(
            signals_df,
            prices_df,
            breakout_days=int(breakout_days),
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            min_samples=int(min_samples),
            confidence_samples=int(confidence_samples),
            quantiles=int(quantiles),
            edge_penalty_scale=float(edge_penalty_scale),
        )

    featured = compute_signal_penalty_features(
        signals_df,
        prices_df,
        breakout_days=int(breakout_days),
        recent_signal_lookback_days=int(selected_recent_signal_lookback_days),
    )
    outcomes = _classify_outcomes(
        featured,
        prices_df,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
    )
    if outcomes.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "recent_signal_lookback_days": int(selected_recent_signal_lookback_days),
            "recent_signal_lookback_diagnostics": lookback_diagnostics,
            "features": {},
        }

    featured = featured.copy()
    featured["signal_date"] = featured["signal_date"].astype(str)
    merged = featured.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    return _build_penalty_payload_from_merged(
        merged,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        min_samples=int(min_samples),
        confidence_samples=int(confidence_samples),
        quantiles=int(quantiles),
        recent_signal_lookback_days=int(selected_recent_signal_lookback_days),
        lookback_diagnostics=lookback_diagnostics,
        edge_penalty_scale=float(edge_penalty_scale),
    )


def compute_penalty_weights_from_training_data(
    training_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    max_hold_days: int,
    min_samples: int,
    confidence_samples: int,
    quantiles: int,
    recent_signal_lookback_days: int,
    edge_penalty_scale: float,
) -> dict:
    selected_recent_signal_lookback_days = int(recent_signal_lookback_days)
    lookback_diagnostics: list[dict] = []
    if selected_recent_signal_lookback_days <= 0:
        selected_recent_signal_lookback_days, lookback_diagnostics = _select_recent_signal_lookback_days_from_training_data(
            training_df,
            target_pct=float(target_pct),
            stop_pct=float(stop_pct),
            max_hold_days=int(max_hold_days),
            min_samples=int(min_samples),
            confidence_samples=int(confidence_samples),
            quantiles=int(quantiles),
            edge_penalty_scale=float(edge_penalty_scale),
        )

    featured = _featured_from_training_data(training_df, lookback_days=int(selected_recent_signal_lookback_days))
    outcomes = _outcomes_from_training_data(training_df)
    featured["signal_date"] = featured["signal_date"].astype(str)
    outcomes["signal_date"] = outcomes["signal_date"].astype(str)
    merged = featured.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    if merged.empty:
        return {
            "computed_at": date.today().isoformat(),
            "signals_analyzed": 0,
            "recent_signal_lookback_days": int(selected_recent_signal_lookback_days),
            "recent_signal_lookback_diagnostics": lookback_diagnostics,
            "features": {},
        }

    return _build_penalty_payload_from_merged(
        merged,
        target_pct=float(target_pct),
        stop_pct=float(stop_pct),
        max_hold_days=int(max_hold_days),
        min_samples=int(min_samples),
        confidence_samples=int(confidence_samples),
        quantiles=int(quantiles),
        recent_signal_lookback_days=int(selected_recent_signal_lookback_days),
        lookback_diagnostics=lookback_diagnostics,
        edge_penalty_scale=float(edge_penalty_scale),
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
        result = compute_penalty_weights_from_training_data(
            training,
            target_pct=args.target_pct,
            stop_pct=args.stop_pct,
            max_hold_days=args.max_hold_days,
            min_samples=args.min_samples,
            confidence_samples=args.confidence_samples,
            quantiles=args.quantiles,
            recent_signal_lookback_days=args.recent_signal_lookback_days,
            edge_penalty_scale=args.edge_penalty_scale,
        )
    else:
        if not prices_path.exists():
            print(f"ERROR: Prices file not found: {prices_path}")
            sys.exit(1)
        if not signals_path.exists():
            print(f"ERROR: Signals file not found: {signals_path}")
            sys.exit(1)

        prices = pd.read_csv(prices_path, parse_dates=["Date"])
        signals = pd.read_csv(signals_path)
        if "signal_date" in signals.columns:
            signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")

        result = compute_penalty_weights(
            signals,
            prices,
            target_pct=args.target_pct,
            stop_pct=args.stop_pct,
            max_hold_days=args.max_hold_days,
            min_samples=args.min_samples,
            confidence_samples=args.confidence_samples,
            quantiles=args.quantiles,
            breakout_days=args.breakout_days,
            recent_signal_lookback_days=args.recent_signal_lookback_days,
            edge_penalty_scale=args.edge_penalty_scale,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Signals analyzed: {result.get('signals_analyzed', 0)}")
    print(f"Saved penalty weights to: {out_path}")


if __name__ == "__main__":
    main()