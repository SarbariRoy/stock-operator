"""Compute candle-shape enhancer weights from historical signal outcomes.

Runs weekly (or on-demand) to analyze which candlestick patterns
(Doji, Hammer, Bullish Marubozu, Morning Star, Bullish Engulfing, Bullish Harami,
Piercing Line, Piercing Variant, Inverted Hammer, Bullish Belt Hold, Three White Soldiers) have a positive
win-rate edge when present at signal dates.

Output: stock_triggers/data/st_lt_candle_weights.json
  {
    "doji": 2.5,
    "hammer": 1.0,
        "marubozu": 2.0,
        "confirmed_hammer_a": 2.0,
    "morning_star": 3.5,
    "engulfing": 2.0,
        "engulfing_trend_combo": 3.0,
    "harami": 1.5,
    "piercing_line": 1.0,
    "piercing_variant": 1.0,
    "piercing_variant_b_combo": 2.0,
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"
DEFAULT_OUTPUT = DATA_DIR / "st_lt_candle_weights.json"

from stock_triggers.ui.enhancers import (  # noqa: E402
    bullish_belt_hold,
    bullish_engulfing,
    bullish_harami,
    bullish_marubozu,
    dragonfly_doji,
    hammer,
    inverted_hammer,
    morning_star,
    piercing_line,
    piercing_variant,
    three_white_soldiers,
)
from stock_triggers.training_utils import add_recency_weights, filter_by_date_window, get_sample_weight_series, parse_optional_date, weighted_mean

ENGULFING_POSITIVE_FAMILIES = {"A", "C", "G"}
PIERCING_VARIANT_POSITIVE_FAMILIES = {"B"}

CHECKS = [
    ("doji", dragonfly_doji.check),
    ("hammer", hammer.check),
    ("marubozu", bullish_marubozu.check),
    ("confirmed_hammer_a", None),
    ("morning_star", morning_star.check),
    ("engulfing", bullish_engulfing.check),
    ("engulfing_trend_combo", None),
    ("harami", bullish_harami.check),
    ("piercing_line", piercing_line.check),
    ("piercing_variant", piercing_variant.check),
    ("piercing_variant_b_combo", None),
    ("inverted_hammer", inverted_hammer.check),
    ("belt_hold", bullish_belt_hold.check),
    ("three_white_soldiers", three_white_soldiers.check),
]

COMPARISON_CHECKS = [
    ("hammer_legacy", hammer.check_basic),
    ("engulfing_confirmed_trial", bullish_engulfing.check_confirmed),
]

FEATURE_KEYS = [name for name, _ in CHECKS]
FAMILY_KEYS = ("A", "B", "C", "D", "E", "F", "G")


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

    with_weights = get_sample_weight_series(with_pat)
    without_weights = get_sample_weight_series(without_pat)
    wr_with = weighted_mean((with_pat["outcome"] == "win").astype(float), with_weights)
    lr_with = weighted_mean((with_pat["outcome"] == "loss").astype(float), with_weights)
    wr_without = weighted_mean((without_pat["outcome"] == "win").astype(float), without_weights) if n_without > 0 else baseline_wr
    lr_without = weighted_mean((without_pat["outcome"] == "loss").astype(float), without_weights) if n_without > 0 else 0.0

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


def _round_signed_weight(raw: float, max_weight: float) -> float:
    clipped = max(-float(max_weight), min(float(max_weight), float(raw)))
    return round(clipped * 2) / 2


def _fit_overlap_weights(
    df: pd.DataFrame,
    *,
    feature_keys: list[str],
    scale: float,
    max_weight: float,
    ridge_alpha: float,
    shrinkage_k: float,
) -> tuple[dict[str, float], dict[str, dict], dict[str, float]]:
    fit_df = df[df["outcome"].isin(["win", "loss"])].copy()
    zero_weights = {key: 0.0 for key in feature_keys}
    zero_details = {
        key: {
            "count": int(df.get(key, pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if key in df.columns else 0,
            "fit_count": int(fit_df.get(key, pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if key in fit_df.columns else 0,
            "coef_pp": 0.0,
            "coef_pp_shrunk": 0.0,
            "shrinkage": 0.0,
            "weight": 0.0,
        }
        for key in feature_keys
    }
    diagnostics = {
        "fit_rows": int(len(fit_df)),
        "ridge_alpha": float(ridge_alpha),
        "shrinkage_k": float(shrinkage_k),
    }
    if fit_df.empty:
        return zero_weights, zero_details, diagnostics

    X = fit_df[feature_keys].fillna(False).astype(float).to_numpy(dtype=float)
    if X.size == 0:
        return zero_weights, zero_details, diagnostics
    y = (fit_df["outcome"] == "win").astype(float).to_numpy(dtype=float)
    sample_weights = get_sample_weight_series(fit_df).to_numpy(dtype=float)

    design = np.column_stack([np.ones(len(fit_df), dtype=float), X])
    reg = np.eye(design.shape[1], dtype=float)
    reg[0, 0] = 0.0  # do not penalize intercept
    sqrt_w = np.sqrt(np.clip(sample_weights, a_min=0.0, a_max=None))
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y * sqrt_w
    beta = np.linalg.pinv(weighted_design.T @ weighted_design + float(ridge_alpha) * reg) @ (weighted_design.T @ weighted_y)
    intercept = float(beta[0])
    coefs = beta[1:]

    feature_counts = fit_df[feature_keys].fillna(False).astype(bool).sum(axis=0)
    weights: dict[str, float] = {}
    details: dict[str, dict] = {}
    for idx, key in enumerate(feature_keys):
        fit_count = int(feature_counts.get(key, 0))
        shrinkage = float(fit_count / (fit_count + float(shrinkage_k))) if fit_count > 0 else 0.0
        coef_pp = float(coefs[idx]) * 100.0
        coef_pp_shrunk = coef_pp * shrinkage
        weight = _round_signed_weight(coef_pp_shrunk * float(scale), max_weight)
        weights[key] = weight
        details[key] = {
            "count": int(df[key].fillna(False).astype(bool).sum()) if key in df.columns else 0,
            "fit_count": fit_count,
            "coef_pp": round(coef_pp, 2),
            "coef_pp_shrunk": round(coef_pp_shrunk, 2),
            "shrinkage": round(shrinkage, 3),
            "weight": weight,
        }

    diagnostics.update(
        {
            "intercept_pp": round(intercept * 100.0, 2),
            "baseline_win_rate": round(weighted_mean(y, sample_weights) * 100.0, 1),
        }
    )
    return weights, details, diagnostics


def _merge_model_into_stats(
    stats_map: dict[str, dict],
    *,
    model_details: dict[str, dict],
    weight_map: dict[str, float],
) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for key, stats in stats_map.items():
        detail = dict(stats.get("details", {}))
        model = model_details.get(key, {})
        detail.update(
            {
                "fit_count": int(model.get("fit_count", 0) or 0),
                "coef_pp": float(model.get("coef_pp", 0.0) or 0.0),
                "coef_pp_shrunk": float(model.get("coef_pp_shrunk", 0.0) or 0.0),
                "shrinkage": float(model.get("shrinkage", 0.0) or 0.0),
                "weight": float(weight_map.get(key, 0.0) or 0.0),
            }
        )
        merged[key] = detail
    return merged


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
    p.add_argument("--ridge-alpha", type=float, default=8.0, help="L2 regularization strength for overlap-aware family fitting")
    p.add_argument("--shrinkage-k", type=float, default=6.0, help="Sample shrinkage factor; larger values pull rare candles harder toward zero")
    p.add_argument("--train-start-date", type=str, default="", help="Only use rows on or after this date (YYYY-MM-DD)")
    p.add_argument("--train-end-date", type=str, default="", help="Only use rows on or before this date (YYYY-MM-DD)")
    p.add_argument("--recency-half-life-months", type=float, default=0.0, help="Half-life in months for recency weighting. 0 disables weighting.")
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
    ridge_alpha: float = 8.0,
    shrinkage_k: float = 6.0,
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
        pat_flags["engulfing_trend_combo"] = bool(
            pat_flags.get("engulfing")
            and str(sig.get("pattern_family", "")).strip().upper() in ENGULFING_POSITIVE_FAMILIES
        )
        pat_flags["piercing_variant_b_combo"] = bool(
            pat_flags.get("piercing_variant")
            and str(sig.get("pattern_family", "")).strip().upper() in PIERCING_VARIANT_POSITIVE_FAMILIES
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

        row = {
            "ticker": ticker,
            "signal_date": str(sd.date()),
            "pattern_family": str(sig.get("pattern_family", "")).strip().upper(),
            "outcome": outcome,
            "sample_weight": float(sig.get("sample_weight", 1.0)) if pd.notna(sig.get("sample_weight", 1.0)) else 1.0,
        }
        row.update(pat_flags)
        rows.append(row)

    if not rows:
        return {
            "doji": 0.0, "hammer": 0.0, "marubozu": 0.0, "confirmed_hammer_a": 0.0, "morning_star": 0.0, "engulfing": 0.0, "engulfing_trend_combo": 0.0, "harami": 0.0, "piercing_line": 0.0,
            "piercing_variant": 0.0, "piercing_variant_b_combo": 0.0,
            "inverted_hammer": 0.0, "belt_hold": 0.0, "three_white_soldiers": 0.0,
            "computed_at": date.today().isoformat(),
            "total_signals": 0,
            "families": {},
            "details": {},
        }

    df = pd.DataFrame(rows)
    if "pattern_family" not in df.columns:
        df["pattern_family"] = ""
    df["pattern_family"] = df["pattern_family"].astype(str).str.strip().str.upper()
    total = len(df)
    sample_weight = get_sample_weight_series(df)
    n_win = int((df["outcome"] == "win").sum())
    n_loss = int((df["outcome"] == "loss").sum())
    n_hold = int((df["outcome"] == "hold").sum())
    baseline_wr = weighted_mean((df["outcome"] == "win").astype(float), sample_weight)

    marginal_stats: dict[str, dict] = {}
    comparison_details: dict[str, dict] = {}
    comparison_summary: dict[str, dict] = {}

    for name, _ in CHECKS:
        marginal_stats[name] = _pattern_stats(
            df,
            name=name,
            baseline_wr=baseline_wr,
            total=total,
            min_samples=min_samples,
            scale=scale,
            max_weight=max_weight,
        )

    global_weights, global_model_details, global_model_diag = _fit_overlap_weights(
        df,
        feature_keys=FEATURE_KEYS,
        scale=scale,
        max_weight=max_weight,
        ridge_alpha=ridge_alpha,
        shrinkage_k=shrinkage_k,
    )
    details = _merge_model_into_stats(
        marginal_stats,
        model_details=global_model_details,
        weight_map=global_weights,
    )

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

    families: dict[str, dict] = {}
    for family in FAMILY_KEYS:
        fam_df = df[df["pattern_family"] == family].copy()
        if fam_df.empty:
            continue
        fam_total = len(fam_df)
        fam_baseline_wr = weighted_mean((fam_df["outcome"] == "win").astype(float), get_sample_weight_series(fam_df)) if fam_total > 0 else 0.0
        fam_stats_map: dict[str, dict] = {}
        for name, _ in CHECKS:
            fam_stats_map[name] = _pattern_stats(
                fam_df,
                name=name,
                baseline_wr=fam_baseline_wr,
                total=fam_total,
                min_samples=min_samples,
                scale=scale,
                max_weight=max_weight,
            )
        fam_weights, fam_model_details, fam_model_diag = _fit_overlap_weights(
            fam_df,
            feature_keys=FEATURE_KEYS,
            scale=scale,
            max_weight=max_weight,
            ridge_alpha=ridge_alpha,
            shrinkage_k=shrinkage_k,
        )
        families[family] = {
            "total_signals": int(fam_total),
            "baseline_win_rate": round(float(fam_baseline_wr) * 100.0, 1),
            "outcomes": {
                "win": int((fam_df["outcome"] == "win").sum()),
                "loss": int((fam_df["outcome"] == "loss").sum()),
                "hold": int((fam_df["outcome"] == "hold").sum()),
            },
            "weights": fam_weights,
            "details": _merge_model_into_stats(
                fam_stats_map,
                model_details=fam_model_details,
                weight_map=fam_weights,
            ),
            "model": fam_model_diag,
        }

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
    engulfing_trial = comparison_summary.get("engulfing_confirmed_trial", {})
    live_engulfing = {
        "count": details.get("engulfing", {}).get("count"),
        "win_rate_with": details.get("engulfing", {}).get("win_rate_with"),
        "loss_rate_with": details.get("engulfing", {}).get("loss_rate_with"),
        "edge_pp": details.get("engulfing", {}).get("edge_pp"),
        "weight": details.get("engulfing", {}).get("weight"),
    }
    engulfing_confirmed_vs_live = {
        "live_count": live_engulfing.get("count"),
        "trial_count": engulfing_trial.get("count"),
        "sample_change": (
            int(engulfing_trial["count"]) - int(live_engulfing["count"])
            if engulfing_trial.get("count") is not None and live_engulfing.get("count") is not None
            else None
        ),
        "live_win_rate_with": live_engulfing.get("win_rate_with"),
        "trial_win_rate_with": engulfing_trial.get("win_rate_with"),
        "win_rate_lift_pp": (
            round(float(engulfing_trial["win_rate_with"]) - float(live_engulfing["win_rate_with"]), 1)
            if engulfing_trial.get("win_rate_with") is not None and live_engulfing.get("win_rate_with") is not None
            else None
        ),
        "live_edge_pp": live_engulfing.get("edge_pp"),
        "trial_edge_pp": engulfing_trial.get("edge_pp"),
        "edge_lift_pp": (
            round(float(engulfing_trial["edge_pp"]) - float(live_engulfing["edge_pp"]), 1)
            if engulfing_trial.get("edge_pp") is not None and live_engulfing.get("edge_pp") is not None
            else None
        ),
        "live_weight": live_engulfing.get("weight"),
        "trial_weight_if_used": engulfing_trial.get("weight"),
    }

    result = {
        **global_weights,
        "computed_at": date.today().isoformat(),
        "total_signals": total,
        "baseline_win_rate": round(baseline_wr * 100, 1),
        "outcomes": {"win": n_win, "loss": n_loss, "hold": n_hold},
        "details": details,
        "families": families,
        "model": {
            "type": "ridge_linear_probability",
            "ridge_alpha": float(ridge_alpha),
            "shrinkage_k": float(shrinkage_k),
            **global_model_diag,
        },
        "comparison_details": comparison_details,
        "comparisons": {
            "hammer_vs_legacy": hammer_vs_legacy,
            "engulfing_confirmed_vs_live": engulfing_confirmed_vs_live,
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
    train_start_date = parse_optional_date(args.train_start_date, arg_name="--train-start-date")
    train_end_date = parse_optional_date(args.train_end_date, arg_name="--train-end-date")
    signals = filter_by_date_window(signals, date_col="signal_date", start_date=train_start_date, end_date=train_end_date)
    if float(args.recency_half_life_months) > 0:
        signals = add_recency_weights(signals, date_col="signal_date", half_life_months=float(args.recency_half_life_months))
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
        ridge_alpha=args.ridge_alpha,
        shrinkage_k=args.shrinkage_k,
    )

    print(f"\n{'='*50}")
    print(f"Baseline win rate: {result['baseline_win_rate']}%")
    print(f"Total signals analyzed: {result['total_signals']}")
    print(f"Outcomes: {result['outcomes']}")
    print(f"\nDerived weights:")
    for name in ("doji", "hammer", "marubozu", "confirmed_hammer_a", "morning_star", "engulfing", "engulfing_trend_combo", "harami", "piercing_line", "piercing_variant", "piercing_variant_b_combo", "inverted_hammer", "belt_hold", "three_white_soldiers"):
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
    engulfing_comparison = result.get("comparisons", {}).get("engulfing_confirmed_vs_live", {})
    if engulfing_comparison:
        print("\nEngulfing trial comparison:")
        print(
            "  live n={live_n} wr={live_wr}% edge={live_edge}pp | "
            "trial n={trial_n} wr={trial_wr}% edge={trial_edge}pp".format(
                live_n=engulfing_comparison.get("live_count", "n/a"),
                live_wr=engulfing_comparison.get("live_win_rate_with", "n/a"),
                live_edge=engulfing_comparison.get("live_edge_pp", "n/a"),
                trial_n=engulfing_comparison.get("trial_count", "n/a"),
                trial_wr=engulfing_comparison.get("trial_win_rate_with", "n/a"),
                trial_edge=engulfing_comparison.get("trial_edge_pp", "n/a"),
            )
        )
        print(
            f"  lift: win_rate={engulfing_comparison.get('win_rate_lift_pp', 'n/a')}pp, "
            f"edge={engulfing_comparison.get('edge_lift_pp', 'n/a')}pp"
        )
    print(f"{'='*50}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
