"""Build persisted default LT/ST page artifacts for fast UI loads.

This script precomputes the default Long Term and ST Backtesting views used by
`stock_triggers/ui/app.py` and writes CSV/JSON artifacts under stock_triggers/data.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scoring_defaults import (  # noqa: E402
    LT_DEFAULT_MIN_SCORE,
    ST_DEFAULT_MIN_SCORE,
    build_scoring_defaults_snapshot,
    compute_scoring_defaults_hash,
)
from stock_triggers.scripts.match_backtesting_lab import (  # noqa: E402
    _apply_lab_stop_mode,
    build_signal_tracker_reinvest_parallel,
)

DATA_DIR = ROOT / "stock_triggers" / "data"
SIGNALS_ALL_PATTERNS_CSV = DATA_DIR / "st_signals_all_patterns.csv"
PRICES_CSV = DATA_DIR / "st_lt_prices_eod.csv"

LT_DEFAULT_VIEW_CSV = DATA_DIR / "lt_default_view.csv"
ST_DEFAULT_VIEW_CSV = DATA_DIR / "st_default_view.csv"
ST_DEFAULT_MONTHLY_CSV = DATA_DIR / "st_default_monthly.csv"
ST_DEFAULT_BUCKET_CSV = DATA_DIR / "st_default_bucket.csv"
DEFAULT_VIEW_ARTIFACT_META_JSON = DATA_DIR / "default_view_artifacts_meta.json"


def _load_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()


def _apply_signal_recency_month_filter(signals_df: pd.DataFrame, months: int) -> pd.DataFrame:
    if signals_df.empty:
        return signals_df.copy()

    selected_months = max(0, int(months or 0))
    if selected_months <= 0:
        return signals_df.copy()

    working = signals_df.copy()
    working["_signal_date_dt"] = pd.to_datetime(working.get("signal_date"), errors="coerce")
    working = working[working["_signal_date_dt"].notna()].copy()
    if working.empty:
        return working.drop(columns=["_signal_date_dt"], errors="ignore")

    anchor_dt = working["_signal_date_dt"].max()
    cutoff_dt = (anchor_dt - pd.DateOffset(months=selected_months)).normalize()
    filtered = working[working["_signal_date_dt"] >= cutoff_dt].copy()
    filtered.drop(columns=["_signal_date_dt"], inplace=True, errors="ignore")
    return filtered


def _summarize_signal_tracker(view: pd.DataFrame) -> dict[str, float | int]:
    if view.empty:
        return {
            "n_total": 0,
            "n_target": 0,
            "n_stop": 0,
            "n_holding": 0,
            "avg_return_pct": 0.0,
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

    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    view_with_dates = view.copy()
    view_with_dates["signal_date_dt"] = pd.to_datetime(view_with_dates.get("signal_date"), errors="coerce")
    recent_holding_mask = (view_with_dates["status"] == "Holding") & (view_with_dates["signal_date_dt"] >= cutoff_date)
    analysis_view = view_with_dates[~recent_holding_mask].copy()

    avg_return_pct = float(pd.to_numeric(analysis_view.get("return_pct"), errors="coerce").mean()) if (not analysis_view.empty and "return_pct" in analysis_view.columns) else 0.0

    total_invested = float(pd.to_numeric(view.get("invested"), errors="coerce").fillna(0.0).sum())
    total_current = float(pd.to_numeric(view.get("current_value"), errors="coerce").fillna(0.0).sum())
    total_pnl = float(pd.to_numeric(view.get("pnl"), errors="coerce").fillna(0.0).sum())
    reinvest_enabled = bool("capital_mode" in view.columns and view["capital_mode"].astype(str).eq("reinvest_parallel").any())
    initial_capital = 0.0
    if reinvest_enabled and "initial_capital" in view.columns:
        init_series = pd.to_numeric(view.get("initial_capital"), errors="coerce").dropna()
        if not init_series.empty:
            initial_capital = float(init_series.iloc[0])
    if reinvest_enabled and initial_capital > 0:
        overall_return = (total_pnl / initial_capital) * 100.0
    else:
        overall_return = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0.0

    closed_view = view[view["status"].isin(["Target Hit ✅", "Stop Hit 🛑"])].copy()
    closed_invested = float(pd.to_numeric(closed_view.get("invested"), errors="coerce").fillna(0.0).sum()) if not closed_view.empty else 0.0
    closed_current = float(pd.to_numeric(closed_view.get("current_value"), errors="coerce").fillna(0.0).sum()) if not closed_view.empty else 0.0
    closed_pnl = float(pd.to_numeric(closed_view.get("pnl"), errors="coerce").fillna(0.0).sum()) if not closed_view.empty else 0.0
    closed_return = ((closed_current / closed_invested) - 1) * 100 if closed_invested > 0 else 0.0
    win_rate = (n_target / (n_target + n_stop) * 100) if (n_target + n_stop) > 0 else 0.0

    return {
        "n_total": int(n_total),
        "n_target": int(n_target),
        "n_stop": int(n_stop),
        "n_holding": int(n_holding),
        "avg_return_pct": float(avg_return_pct),
        "total_invested": float(total_invested),
        "total_current": float(total_current),
        "total_pnl": float(total_pnl),
        "overall_return": float(overall_return),
        "closed_invested": float(closed_invested),
        "closed_current": float(closed_current),
        "closed_pnl": float(closed_pnl),
        "closed_return": float(closed_return),
        "win_rate": float(win_rate),
    }


def _summarize_signal_tracker_monthly(view: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    columns = [
        "month",
        "start_capital",
        "trades",
        "invested",
        "recycled_capital",
        "idle_cash",
        "utilization_%",
        "current_value",
        "return_value",
        "return_pct",
        "pool_return_pct",
        "avg_trade_return_pct",
        "end_capital",
    ]
    empty_stats: dict[str, float | int] = {
        "months": 0,
        "avg_monthly_invested": 0.0,
        "avg_monthly_return_value": 0.0,
        "avg_monthly_return_pct": 0.0,
    }
    if view.empty or "signal_date" not in view.columns:
        return pd.DataFrame(columns=columns), empty_stats

    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    monthly_base = view.copy()
    monthly_base["signal_date_temp"] = pd.to_datetime(monthly_base.get("signal_date"), errors="coerce")
    recent_holding = (monthly_base.get("status", "") == "Holding") & (monthly_base["signal_date_temp"] >= cutoff_date)
    monthly_base = monthly_base[~recent_holding].copy()
    if monthly_base.empty and not view.empty:
        return pd.DataFrame(columns=columns), empty_stats

    monthly = monthly_base.copy()
    monthly["signal_date"] = pd.to_datetime(monthly["signal_date"], errors="coerce")
    monthly = monthly.dropna(subset=["signal_date"]).copy()
    if monthly.empty:
        return pd.DataFrame(columns=columns), empty_stats

    for col in ("invested", "current_value", "pnl", "return_pct"):
        if col in monthly.columns:
            monthly[col] = pd.to_numeric(monthly[col], errors="coerce").fillna(0.0)
        else:
            monthly[col] = 0.0

    monthly["month"] = monthly["signal_date"].dt.to_period("M").astype(str)

    use_pool_column = "capital_pool_before" in monthly.columns
    if use_pool_column:
        monthly["capital_pool_before"] = pd.to_numeric(monthly["capital_pool_before"], errors="coerce")
        pool_open = (
            monthly.sort_values("signal_date")
            .groupby("month", as_index=False)["capital_pool_before"]
            .first()
            .rename(columns={"capital_pool_before": "_pool_open"})
        )

    grouped = (
        monthly.groupby("month", as_index=False)
        .agg(
            trades=("signal_date", "size"),
            invested=("invested", "sum"),
            current_value=("current_value", "sum"),
            return_value=("pnl", "sum"),
            avg_trade_return_pct=("return_pct", "mean"),
        )
        .sort_values("month", ascending=True)
        .reset_index(drop=True)
    )

    if use_pool_column:
        grouped = grouped.merge(pool_open, on="month", how="left")

    initial_capital = 0.0
    if "initial_capital" in monthly.columns:
        ic_series = pd.to_numeric(monthly["initial_capital"], errors="coerce").dropna()
        if not ic_series.empty:
            initial_capital = float(ic_series.iloc[0])
    if initial_capital <= 0.0:
        first_month_invested = float(grouped["invested"].iloc[0]) if not grouped.empty else 0.0
        initial_capital = first_month_invested if first_month_invested > 0 else 1.0

    start_capitals: list[float] = []
    end_capitals: list[float] = []
    idle_cash_list: list[float] = []
    recycled_list: list[float] = []
    utilization_list: list[float] = []
    running = initial_capital
    for row in grouped.itertuples(index=False):
        pool_val = getattr(row, "_pool_open", float("nan")) if use_pool_column else float("nan")
        month_start = float(pool_val) if not pd.isna(float(pool_val)) else running
        start_capitals.append(round(month_start, 2))
        inv = float(row.invested)
        recycled = round(max(inv - month_start, 0.0), 2)
        idle = round(max(month_start - inv, 0.0), 2)
        utilization = round(inv / month_start * 100.0, 1) if month_start > 0 else 0.0
        recycled_list.append(recycled)
        idle_cash_list.append(idle)
        utilization_list.append(utilization)
        end_cap = round(month_start + float(row.return_value), 2)
        end_capitals.append(end_cap)
        running = end_cap

    grouped["start_capital"] = start_capitals
    grouped["recycled_capital"] = recycled_list
    grouped["idle_cash"] = idle_cash_list
    grouped["utilization_%"] = utilization_list
    grouped["end_capital"] = end_capitals
    grouped["return_pct"] = grouped.apply(
        lambda row: round(float(row["return_value"]) / float(row["invested"]) * 100.0, 2) if float(row["invested"]) > 0 else 0.0,
        axis=1,
    )
    grouped["pool_return_pct"] = grouped.apply(
        lambda row: round(float(row["return_value"]) / float(row["start_capital"]) * 100.0, 2) if float(row["start_capital"]) > 0 else 0.0,
        axis=1,
    )
    grouped = grouped.sort_values("month", ascending=False)

    numeric_cols = [
        "start_capital",
        "invested",
        "recycled_capital",
        "idle_cash",
        "current_value",
        "return_value",
        "return_pct",
        "pool_return_pct",
        "avg_trade_return_pct",
        "end_capital",
    ]
    grouped[numeric_cols] = grouped[numeric_cols].round(2)
    grouped = grouped.drop(columns=["_pool_open"], errors="ignore")

    monthly_return_series = grouped["current_value"] - grouped["invested"] if not grouped.empty else pd.Series(dtype=float)
    stats: dict[str, float | int] = {
        "months": int(len(grouped)),
        "avg_monthly_invested": float(grouped["invested"].mean()) if not grouped.empty else 0.0,
        "avg_monthly_return_value": float(monthly_return_series.mean()) if not grouped.empty else 0.0,
        "avg_monthly_return_pct": float(grouped["return_pct"].mean()) if not grouped.empty else 0.0,
    }
    return grouped[columns], stats


def _summarize_score_bucket_win_rates(view: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    columns = ["score_bucket", "signals", "closed", "target_hit", "stop_hit", "holding", "win_rate_pct", "avg_return_pct"]
    bucket_labels = [f"{start}-{start + 10}" for start in range(0, 100, 10)]
    if view.empty or score_col not in view.columns:
        return pd.DataFrame(
            {
                "score_bucket": bucket_labels,
                "signals": [0] * len(bucket_labels),
                "closed": [0] * len(bucket_labels),
                "target_hit": [0] * len(bucket_labels),
                "stop_hit": [0] * len(bucket_labels),
                "holding": [0] * len(bucket_labels),
                "win_rate_pct": [0.0] * len(bucket_labels),
                "avg_return_pct": [0.0] * len(bucket_labels),
            }
        )[columns]

    working = view.copy()
    working["_score_bucket_value"] = pd.to_numeric(working.get(score_col), errors="coerce")
    working = working[working["_score_bucket_value"].notna()].copy()
    if working.empty:
        return pd.DataFrame(
            {
                "score_bucket": bucket_labels,
                "signals": [0] * len(bucket_labels),
                "closed": [0] * len(bucket_labels),
                "target_hit": [0] * len(bucket_labels),
                "stop_hit": [0] * len(bucket_labels),
                "holding": [0] * len(bucket_labels),
                "win_rate_pct": [0.0] * len(bucket_labels),
                "avg_return_pct": [0.0] * len(bucket_labels),
            }
        )[columns]

    bucket_edges = list(range(0, 100, 10)) + [101]
    working["score_bucket"] = pd.cut(
        working["_score_bucket_value"].clip(lower=0.0, upper=100.0),
        bins=bucket_edges,
        labels=bucket_labels,
        right=False,
        include_lowest=True,
    )
    working["_is_target"] = (working["status"] == "Target Hit ✅").astype(int)
    working["_is_stop"] = (working["status"] == "Stop Hit 🛑").astype(int)
    working["_is_holding"] = (working["status"] == "Holding").astype(int)
    working["_is_closed"] = working["status"].isin(["Target Hit ✅", "Stop Hit 🛑"]).astype(int)
    working["return_pct"] = pd.to_numeric(working.get("return_pct"), errors="coerce")

    grouped = (
        working.groupby("score_bucket", observed=False, as_index=False)
        .agg(
            signals=("score_bucket", "size"),
            closed=("_is_closed", "sum"),
            target_hit=("_is_target", "sum"),
            stop_hit=("_is_stop", "sum"),
            holding=("_is_holding", "sum"),
            avg_return_pct=("return_pct", "mean"),
        )
    )
    grouped["win_rate_pct"] = grouped.apply(
        lambda row: round(float(row["target_hit"]) / float(row["closed"]) * 100.0, 1) if float(row["closed"]) > 0 else 0.0,
        axis=1,
    )
    grouped["avg_return_pct"] = pd.to_numeric(grouped["avg_return_pct"], errors="coerce").fillna(0.0).round(2)

    template = pd.DataFrame({"score_bucket": bucket_labels})
    grouped = template.merge(grouped, on="score_bucket", how="left")
    for col in ["signals", "closed", "target_hit", "stop_hit", "holding"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0).astype(int)
    for col in ["win_rate_pct", "avg_return_pct"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0).round(2)
    return grouped[columns]


def _apply_st_structure_confluence_stop(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    fixed_stop_pct: float = 2.0,
    structure_lookback: int = 10,
    structure_buffer_pct: float = 0.5,
    ema_period: int = 20,
    vwap_period: int = 20,
) -> pd.DataFrame:
    if signals_df.empty or prices_df.empty:
        return signals_df.copy()

    out = signals_df.copy()
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip().str.removesuffix(".NS")
    grouped = {str(ticker): grp.sort_values("Date").copy() for ticker, grp in prices.groupby("Ticker", sort=False)}

    for idx in out.index:
        ticker = str(out.at[idx, "ticker"]).upper().strip().removesuffix(".NS")
        hist = grouped.get(ticker)
        entry_price = float(pd.to_numeric(out.at[idx, "entry_price"], errors="coerce") or 0.0)
        fallback_stop = entry_price * (1.0 - float(fixed_stop_pct) / 100.0)
        if hist is None or entry_price <= 0:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        sig_date = pd.to_datetime(out.at[idx, "signal_date"], errors="coerce")
        hist = hist[hist["Date"] <= sig_date].copy()
        if hist.empty:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        recent_low = float(hist["Low"].tail(int(structure_lookback)).min()) if "Low" in hist.columns else None

        ema20_value = None
        if "Close" in hist.columns:
            ema_series = pd.to_numeric(hist["Close"], errors="coerce").ewm(span=int(ema_period), adjust=False).mean()
            if not ema_series.empty and pd.notna(ema_series.iloc[-1]):
                ema20_value = float(ema_series.iloc[-1])

        vwap20_value = None
        if "Close" in hist.columns and "Volume" in hist.columns:
            tail = hist.tail(int(vwap_period)).copy()
            px = pd.to_numeric(tail["Close"], errors="coerce")
            vol = pd.to_numeric(tail["Volume"], errors="coerce")
            vol_sum = float(vol.sum()) if not vol.empty else 0.0
            if vol_sum > 0:
                vwap20_value = float((px * vol).sum() / vol_sum)

        anchor_candidates = [
            value for value in [recent_low, ema20_value, vwap20_value]
            if value is not None and pd.notna(value) and float(value) > 0
        ]
        if anchor_candidates:
            stop_price = min(float(value) for value in anchor_candidates) * (1.0 - float(structure_buffer_pct) / 100.0)
            stop_price = max(float(stop_price), float(entry_price) * 0.90)
        else:
            stop_price = fallback_stop

        if stop_price <= 0 or stop_price >= entry_price:
            stop_price = fallback_stop

        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0 if entry_price > 0 else float(fixed_stop_pct)
        out.at[idx, "stop_price"] = round(float(stop_price), 4)
        out.at[idx, "stop_pct"] = round(float(stop_pct_eff), 2)

    return out


def _build_lt_default_view(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    lt_signals = signals_df.copy()
    if lt_signals.empty or prices_df.empty:
        return pd.DataFrame(), _summarize_signal_tracker(pd.DataFrame())

    lt_signals = _apply_signal_recency_month_filter(lt_signals, 24)
    lt_signals = _apply_lab_stop_mode(
        lt_signals,
        prices_df,
        stop_mode="structure_atr",
        fixed_stop_pct=9.0,
        atr_period=14,
        atr_multiplier=2.5,
        structure_lookback=5,
        structure_atr_buffer=2.5,
    )
    lt_signals["signal_score"] = pd.to_numeric(lt_signals.get("signal_score"), errors="coerce").fillna(0.0)
    lt_signals = lt_signals[lt_signals["signal_score"] >= float(LT_DEFAULT_MIN_SCORE)].copy()

    tracker = build_signal_tracker_reinvest_parallel(
        lt_signals,
        prices_df,
        target_pct=6.0,
        stop_pct=9.0,
        initial_capital=10000.0,
    )
    if tracker.empty:
        return tracker, _summarize_signal_tracker(tracker)

    tracker["days_held"] = pd.to_numeric(tracker.get("days_held"), errors="coerce")
    view = tracker[tracker["days_held"].fillna(10**9) <= 60].copy()
    if "signal_score" in view.columns:
        view.sort_values(["signal_score", "signal_date", "ticker"], ascending=[False, False, True], inplace=True)
    else:
        view.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)
    return view, _summarize_signal_tracker(view)


def _build_st_default_view(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    st_signals = signals_df.copy()
    if st_signals.empty or prices_df.empty:
        empty_view = pd.DataFrame()
        empty_summary = _summarize_signal_tracker(empty_view)
        empty_monthly, empty_monthly_stats = _summarize_signal_tracker_monthly(empty_view)
        empty_bucket = _summarize_score_bucket_win_rates(empty_view, score_col="st_score")
        return empty_view, empty_summary, empty_monthly, empty_monthly_stats, empty_bucket

    st_signals = _apply_signal_recency_month_filter(st_signals, 24)

    score_series = pd.to_numeric(st_signals.get("st_score"), errors="coerce")
    signal_dates = pd.to_datetime(st_signals.get("signal_date"), errors="coerce")
    recent_mask = signal_dates >= (pd.Timestamp.now() - pd.Timedelta(days=7))
    st_signals = st_signals[(score_series >= float(ST_DEFAULT_MIN_SCORE)) | recent_mask].copy()

    st_signals = _apply_st_structure_confluence_stop(st_signals, prices_df, fixed_stop_pct=2.0)

    tracker = build_signal_tracker_reinvest_parallel(
        st_signals,
        prices_df,
        target_pct=3.0,
        stop_pct=2.0,
        initial_capital=10000.0,
    )
    if tracker.empty:
        empty_monthly, empty_monthly_stats = _summarize_signal_tracker_monthly(tracker)
        empty_bucket = _summarize_score_bucket_win_rates(tracker, score_col="st_score")
        return tracker, _summarize_signal_tracker(tracker), empty_monthly, empty_monthly_stats, empty_bucket

    tracker["days_held"] = pd.to_numeric(tracker.get("days_held"), errors="coerce")
    view = tracker[tracker["days_held"].fillna(10**9) <= 7].copy()
    if "signal_date" in view.columns:
        view.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)

    summary = _summarize_signal_tracker(view)
    monthly_view, monthly_stats = _summarize_signal_tracker_monthly(view)
    bucket_view = _summarize_score_bucket_win_rates(view, score_col="st_score")
    return view, summary, monthly_view, monthly_stats, bucket_view


def _source_mtimes(paths: list[Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in paths:
        try:
            out[path.name] = int(path.stat().st_mtime_ns)
        except OSError:
            out[path.name] = 0
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    signals_df = _load_csv(SIGNALS_ALL_PATTERNS_CSV)
    prices_df = _load_csv(PRICES_CSV, parse_dates=["Date"])

    lt_view, lt_summary = _build_lt_default_view(signals_df, prices_df)
    st_view, st_summary, st_monthly_view, st_monthly_stats, st_bucket_view = _build_st_default_view(signals_df, prices_df)

    _write_csv(lt_view, LT_DEFAULT_VIEW_CSV)
    _write_csv(st_view, ST_DEFAULT_VIEW_CSV)
    _write_csv(st_monthly_view, ST_DEFAULT_MONTHLY_CSV)
    _write_csv(st_bucket_view, ST_DEFAULT_BUCKET_CSV)

    defaults_snapshot = build_scoring_defaults_snapshot()
    defaults_hash = compute_scoring_defaults_hash(defaults_snapshot)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_defaults_hash": defaults_hash,
        "source_mtimes_ns": _source_mtimes([SIGNALS_ALL_PATTERNS_CSV, PRICES_CSV]),
        "lt": {
            "view_path": LT_DEFAULT_VIEW_CSV.name,
            "rows": int(len(lt_view)),
            "summary": lt_summary,
            "defaults": {
                "target_pct": 6.0,
                "stop_pct": 9.0,
                "min_score": int(LT_DEFAULT_MIN_SCORE),
                "max_days": 60,
                "recency_months": 24,
                "stop_mode": "Structure + ATR",
                "capital_mode": "Reinvest (parallel allocation)",
                "initial_capital": 10000.0,
                "catalyst_mode": "baseline",
            },
        },
        "st": {
            "view_path": ST_DEFAULT_VIEW_CSV.name,
            "monthly_path": ST_DEFAULT_MONTHLY_CSV.name,
            "bucket_path": ST_DEFAULT_BUCKET_CSV.name,
            "rows": int(len(st_view)),
            "summary": st_summary,
            "monthly_stats": st_monthly_stats,
            "defaults": {
                "target_pct": 3.0,
                "stop_pct": 2.0,
                "min_score": int(ST_DEFAULT_MIN_SCORE),
                "max_days": 7,
                "recency_months": 24,
                "stop_mode": "Structure confluence",
                "capital_mode": "Reinvest (parallel allocation)",
                "initial_capital": 10000.0,
                "catalyst_mode": "baseline",
            },
        },
    }

    DEFAULT_VIEW_ARTIFACT_META_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_VIEW_ARTIFACT_META_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("DEFAULT_VIEW_ARTIFACTS_COMPLETE")
    print(f"LT rows: {len(lt_view)}")
    print(f"ST rows: {len(st_view)}")
    print(f"Meta: {DEFAULT_VIEW_ARTIFACT_META_JSON}")


if __name__ == "__main__":
    main()
