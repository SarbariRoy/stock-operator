"""Reproduce Long Term metrics outside the Streamlit UI.

This script matches the advanced Long Term workflow closely:
- walk-forward/holdout out-of-sample scoping from stop-risk evaluation
- lab stop-mode adjustments
- optional hold-to-target behavior for score > 90 / > 95 stop modes
- catalyst mode reranking/filtering
- tracker-style mark-to-market return calculations
- summary metrics identical in shape to the UI cards

Usage:
  python match_backtesting_lab.py
  python match_backtesting_lab.py --min-score 90 --stop-mode score_gt_90_hold_to_target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.evaluate_stop_risk_walk_forward import (  # noqa: E402
    CANDIDATE_SPECS,
    _compute_labels,
    _load_signals,
    evaluate_candidate,
)
from stock_triggers.ui.patterns.catalyst_ui import CATALYST_MODES, filter_signals_by_catalyst_mode  # noqa: E402

DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_SIGNALS = DATA_DIR / "st_signals_all_patterns.csv"
DEFAULT_PRICES = DATA_DIR / "st_lt_prices_eod.csv"
DEFAULT_PREDICTIONS_COMPARE = DATA_DIR / "lt_stop_risk_walk_forward_predictions_compare.csv"
DEFAULT_SUMMARY_OUT = DATA_DIR / "lt_match_summary.csv"
DEFAULT_VIEW_OUT = DATA_DIR / "lt_match_view.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match Long Term metrics outside the UI.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS), help="Path to st_signals_all_patterns.csv")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Path to st_lt_prices_eod.csv")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(DEFAULT_PREDICTIONS_COMPARE),
        help="Optional cached walk-forward predictions CSV",
    )
    parser.add_argument("--evaluation-mode", choices=["walk-forward", "holdout"], default="walk-forward")
    parser.add_argument("--train-end-date", type=str, default="", help="Holdout cutoff YYYY-MM-DD")
    parser.add_argument("--eval-hold-days", type=int, default=30, help="Hold horizon used for OOS scoping")
    parser.add_argument("--target-pct", type=float, default=6.0)
    parser.add_argument("--stop-pct", type=float, default=9.0)
    parser.add_argument(
        "--stop-mode",
        choices=["structure_atr", "atr", "fixed_pct", "score_gt_95_hold_to_target", "score_gt_90_hold_to_target"],
        default="structure_atr",
    )
    parser.add_argument("--capital-per-trade", type=float, default=10000.0)
    parser.add_argument("--capital-mode", choices=["fixed_per_trade", "reinvest_parallel"], default="fixed_per_trade")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--min-score", type=float, default=90.0)
    parser.add_argument("--max-days-held", type=int, default=60)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-mult", type=float, default=2.5)
    parser.add_argument("--structure-lookback", type=int, default=5)
    parser.add_argument("--catalyst-mode", choices=list(CATALYST_MODES.keys()), default="baseline")
    parser.add_argument("--summary-out", type=str, default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--view-out", type=str, default=str(DEFAULT_VIEW_OUT))
    parser.add_argument("--yearly-out", type=str, default="", help="Optional yearly reinvest summary CSV output")
    return parser.parse_args()


def _normalize_stop_mode(stop_mode: str) -> str:
    value = str(stop_mode or "").strip().lower().replace(" ", "_")
    aliases = {
        "fixed_%": "fixed_pct",
        "fixed": "fixed_pct",
        "atr": "atr",
        "structure+atr": "structure_atr",
        "structure_atr": "structure_atr",
        "score>95holdtotarget": "score_gt_95_hold_to_target",
        "score_gt_95_hold_to_target": "score_gt_95_hold_to_target",
        "score>90holdtotarget": "score_gt_90_hold_to_target",
        "score_gt_90_hold_to_target": "score_gt_90_hold_to_target",
    }
    return aliases.get(value, value)


def _annotate_hold_to_target_only(signals_df: pd.DataFrame, stop_mode: str) -> pd.DataFrame:
    out = signals_df.copy()
    out["hold_to_target_only"] = False
    normalized = _normalize_stop_mode(stop_mode)
    if normalized == "score_gt_95_hold_to_target":
        threshold = 95.0
    elif normalized == "score_gt_90_hold_to_target":
        threshold = 90.0
    else:
        return out
    if out.empty or "signal_score" not in out.columns:
        return out
    scores = pd.to_numeric(out["signal_score"], errors="coerce")
    out.loc[scores > threshold, "hold_to_target_only"] = True
    return out


def _stop_exit_allowed(signal_date: pd.Timestamp, bar_date: pd.Timestamp, *, lockout_days: int = 7) -> bool:
    return (pd.to_datetime(bar_date) - pd.to_datetime(signal_date)).days > int(lockout_days)


def _apply_lab_stop_mode(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    stop_mode: str,
    fixed_stop_pct: float,
    atr_period: int,
    atr_multiplier: float,
    structure_lookback: int,
    structure_atr_buffer: float,
) -> pd.DataFrame:
    if signals_df.empty or prices_df.empty:
        return signals_df.copy()

    out = signals_df.copy()
    px = prices_df.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in px.groupby("Ticker", sort=False)}
    effective_stop_mode = _normalize_stop_mode(stop_mode)

    for idx in out.index:
        ticker = str(out.at[idx, "ticker"])
        ticker_alt = ticker[:-3] if ticker.endswith(".NS") else ticker + ".NS"
        hist = grouped.get(ticker)
        if hist is None:
            hist = grouped.get(ticker_alt)

        entry_price = float(out.at[idx, "entry_price"])
        fallback_stop = entry_price * (1.0 - float(fixed_stop_pct) / 100.0)
        if hist is None or entry_price <= 0:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        sig_date = pd.to_datetime(out.at[idx, "signal_date"])
        hist = hist[hist["Date"] <= sig_date].copy()
        if hist.empty:
            out.at[idx, "stop_price"] = round(fallback_stop, 4)
            out.at[idx, "stop_pct"] = round(float(fixed_stop_pct), 2)
            continue

        tr1 = hist["High"] - hist["Low"]
        tr2 = (hist["High"] - hist["Close"].shift(1)).abs()
        tr3 = (hist["Low"] - hist["Close"].shift(1)).abs()
        hist["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        hist["ATR"] = hist["TR"].rolling(int(atr_period)).mean()
        atr_value = float(hist.iloc[-1]["ATR"]) if pd.notna(hist.iloc[-1]["ATR"]) else None

        if effective_stop_mode == "atr" and atr_value is not None:
            stop_price = entry_price - atr_value * float(atr_multiplier)
        elif effective_stop_mode == "structure_atr" and atr_value is not None:
            recent_low = float(hist["Low"].tail(int(structure_lookback)).min())
            stop_price = recent_low - atr_value * float(structure_atr_buffer)
        else:
            stop_price = fallback_stop

        if effective_stop_mode in {"atr", "structure_atr"}:
            stop_price = max(float(stop_price), float(fallback_stop))

        if stop_price <= 0 or stop_price >= entry_price:
            stop_price = fallback_stop

        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0
        out.at[idx, "stop_price"] = round(float(stop_price), 4)
        out.at[idx, "stop_pct"] = round(float(stop_pct_eff), 2)

    return out


def build_signal_tracker(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    capital_per_trade: float,
) -> pd.DataFrame:
    if signals_df.empty or prices_df.empty:
        return pd.DataFrame()

    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    
    rows: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker_raw = str(sig["ticker"]).upper().strip()
        ticker_norm = ticker_raw.removesuffix(".NS")
        sig_date = pd.to_datetime(sig["signal_date"], errors="coerce")
        
        if pd.isna(sig_date):
            continue
            
        entry_price = float(sig["entry_price"])
        stop_price_sig = float(sig.get("stop_price", entry_price * (1.0 - stop_pct / 100.0)))
        target_price = entry_price * (1.0 + target_pct / 100.0)
        stop_price_calc = stop_price_sig
        hold_to_target_only = bool(sig.get("hold_to_target_only", False))

        qty = int(capital_per_trade // entry_price) if entry_price > 0 else 0
        if qty == 0:
            continue
        invested = round(qty * entry_price, 2)

        future = prices[(prices["Ticker"].str.removesuffix(".NS") == ticker_norm) & (prices["Date"] > sig_date)].sort_values("Date")

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
            if not hold_to_target_only and _stop_exit_allowed(sig_date, bar["Date"]) and close <= stop_price_calc:
                status = "Stop Hit 🛑"
                exit_date = bar["Date"]
                exit_price = stop_price_calc
                break

        if exit_price is not None:
            current_val = round(qty * exit_price, 2)
        else:
            current_val = round(qty * latest_close, 2)

        pnl = round(current_val - invested, 2)
        return_pct = round(((current_val / invested) - 1) * 100, 2) if invested > 0 else 0.0

        if exit_date is not None:
            days_held = (pd.to_datetime(exit_date) - sig_date).days
        elif not future.empty:
            days_held = (future["Date"].max() - sig_date).days
        else:
            days_held = 0

        rows.append(
            {
                "signal_date": sig_date.date().isoformat(),
                "ticker": ticker_norm,
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
                "exit_date": exit_date.date().isoformat() if exit_date is not None and hasattr(exit_date, "date") else (str(exit_date)[:10] if exit_date else "-"),
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
    
    # Exclude only RECENT Holding trades (< 7 days old)
    # Older Holding trades and all closed trades are included in analysis
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    view_with_dates = view.copy()
    view_with_dates["signal_date_dt"] = pd.to_datetime(view_with_dates.get("signal_date"), errors="coerce")
    
    recent_holding_mask = (view_with_dates["status"] == "Holding") & (view_with_dates["signal_date_dt"] >= cutoff_date)
    analysis_view = view_with_dates[~recent_holding_mask].copy()
    
    # Portfolio tracking includes all trades (including all holding)
    total_invested = float(view["invested"].sum())
    total_current = float(view["current_value"].sum())
    total_pnl = float(view["pnl"].sum())
    overall_return = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0.0

    # Closed trades and analysis view metrics (excludes recent Holding only)
    closed_view = view[view["status"].isin(["Target Hit ✅", "Stop Hit 🛑"])].copy()
    closed_invested = float(closed_view["invested"].sum()) if not closed_view.empty else 0.0
    closed_current = float(closed_view["current_value"].sum()) if not closed_view.empty else 0.0
    closed_pnl = float(closed_view["pnl"].sum()) if not closed_view.empty else 0.0
    closed_return = ((closed_current / closed_invested) - 1) * 100 if closed_invested > 0 else 0.0
    win_rate = (n_target / (n_target + n_stop) * 100) if (n_target + n_stop) > 0 else 0.0

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


def build_signal_tracker_reinvest_parallel(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    target_pct: float,
    stop_pct: float,
    initial_capital: float,
) -> pd.DataFrame:
    if signals_df.empty or prices_df.empty:
        return pd.DataFrame()

    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()

    outcomes: list[dict] = []
    for _, sig in signals_df.iterrows():
        ticker_raw = str(sig.get("ticker", "")).upper().strip()
        ticker = ticker_raw.removesuffix(".NS")
        sig_date = pd.to_datetime(sig.get("signal_date"), errors="coerce")
        entry_price = float(sig.get("entry_price", 0.0) or 0.0)
        if pd.isna(sig_date) or entry_price <= 0:
            continue

        stop_price_sig = float(sig.get("stop_price", entry_price * (1.0 - stop_pct / 100.0)))
        target_price = entry_price * (1.0 + target_pct / 100.0)
        hold_to_target_only = bool(sig.get("hold_to_target_only", False))

        future = prices[(prices["Ticker"].str.removesuffix(".NS") == ticker) & (prices["Date"] > sig_date)].sort_values("Date")

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
                exit_date = pd.to_datetime(bar["Date"], errors="coerce")
                exit_price = target_price
                break
            if not hold_to_target_only and _stop_exit_allowed(sig_date, bar["Date"]) and close <= stop_price_sig:
                status = "Stop Hit 🛑"
                exit_date = pd.to_datetime(bar["Date"], errors="coerce")
                exit_price = stop_price_sig
                break

        effective_price = float(exit_price) if exit_price is not None else float(latest_close)
        if exit_date is not None:
            days_held = int((exit_date - sig_date).days)
        elif not future.empty:
            days_held = int((future["Date"].max() - sig_date).days)
        else:
            days_held = 0

        outcomes.append({
            "signal_date": sig_date,
            "ticker": ticker,
            "pattern": str(sig.get("pattern", "")),
            "pattern_family": str(sig.get("pattern_family", "")),
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_price": stop_price_sig,
            "latest_close": latest_close,
            "effective_price": effective_price,
            "status": status,
            "exit_date": exit_date,
            "days_held": days_held,
            "hold_to_target_only": hold_to_target_only,
            "signal_score": round(float(sig["signal_score"]), 1) if pd.notna(sig.get("signal_score")) else None,
        })

    if not outcomes:
        return pd.DataFrame()

    outcomes = sorted(outcomes, key=lambda r: (r["signal_date"], str(r["ticker"]), str(r["pattern"])))

    available_cash = float(initial_capital)
    open_positions: list[dict] = []
    rows: list[dict] = []

    idx = 0
    while idx < len(outcomes):
        day = pd.to_datetime(outcomes[idx]["signal_date"], errors="coerce").normalize()

        pending: list[dict] = []
        for pos in open_positions:
            release_date = pos.get("release_date")
            if release_date is not None and pd.to_datetime(release_date, errors="coerce").normalize() <= day:
                available_cash += float(pos["current_value"])
            else:
                pending.append(pos)
        open_positions = pending

        day_start = idx
        while idx < len(outcomes) and pd.to_datetime(outcomes[idx]["signal_date"], errors="coerce").normalize() == day:
            idx += 1
        day_items = outcomes[day_start:idx]
        day_cash_before = float(available_cash)
        per_trade_budget = day_cash_before / float(len(day_items)) if day_items else 0.0

        for item in day_items:
            qty = int(float(per_trade_budget) // float(item["entry_price"])) if float(item["entry_price"]) > 0 else 0
            if qty <= 0:
                continue
            invested = round(qty * float(item["entry_price"]), 2)
            available_cash -= invested
            current_val = round(qty * float(item["effective_price"]), 2)
            pnl = round(current_val - invested, 2)
            return_pct = round(((current_val / invested) - 1.0) * 100.0, 2) if invested > 0 else 0.0

            open_positions.append({
                "release_date": item["exit_date"] if item["exit_date"] is not None else None,
                "current_value": current_val,
            })

            rows.append(
                {
                    "signal_date": item["signal_date"].date().isoformat(),
                    "ticker": item["ticker"],
                    "pattern": item["pattern"],
                    "pattern_family": item["pattern_family"],
                    "entry_price": round(float(item["entry_price"]), 2),
                    "qty": qty,
                    "invested": invested,
                    "target_price": round(float(item["target_price"]), 2),
                    "stop_price": round(float(item["stop_price"]), 2),
                    "latest_close": round(float(item["latest_close"]), 2),
                    "current_value": current_val,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "days_held": int(item["days_held"]),
                    "exit_date": item["exit_date"].date().isoformat() if item["exit_date"] is not None and hasattr(item["exit_date"], "date") else "-",
                    "status": item["status"],
                    "signal_score": item["signal_score"],
                    "hold_to_target_only": item["hold_to_target_only"],
                    "capital_mode": "reinvest_parallel",
                    "initial_capital": float(initial_capital),
                    "allocated_capital": round(float(per_trade_budget), 2),
                    "capital_pool_before": round(float(day_cash_before), 2),
                    "capital_pool_after_entry": round(float(available_cash), 2),
                    "final_capital_snapshot": None,
                }
            )

    final_capital = float(available_cash) + sum(float(pos.get("current_value", 0.0)) for pos in open_positions)
    for row in rows:
        row["final_capital_snapshot"] = round(final_capital, 2)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["signal_date", "ticker"], ascending=[False, True], inplace=True)
    return out


def summarize_reinvest_yearly(view: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "starting_capital", "realized_pnl", "ending_capital", "realized_return_pct"]
    if view.empty or "capital_mode" not in view.columns:
        return pd.DataFrame(columns=cols)

    reinvest = view[view["capital_mode"].astype(str).eq("reinvest_parallel")].copy()
    if reinvest.empty:
        return pd.DataFrame(columns=cols)

    closed = reinvest[reinvest["status"].isin(["Target Hit ✅", "Stop Hit 🛑"])].copy()
    closed["exit_dt"] = pd.to_datetime(closed["exit_date"], errors="coerce")
    closed = closed[closed["exit_dt"].notna()].copy()
    if closed.empty:
        return pd.DataFrame(columns=cols)

    closed["year"] = closed["exit_dt"].dt.year.astype(int)
    yearly = closed.groupby("year", as_index=False)["pnl"].sum().rename(columns={"pnl": "realized_pnl"})

    init_series = pd.to_numeric(reinvest.get("initial_capital"), errors="coerce").dropna()
    running = float(init_series.iloc[0]) if not init_series.empty else 0.0

    rows: list[dict] = []
    for _, rec in yearly.sort_values("year").iterrows():
        start = float(running)
        pnl = float(rec["realized_pnl"])
        end = float(start + pnl)
        ret = ((end / start) - 1.0) * 100.0 if start > 0 else 0.0
        rows.append(
            {
                "year": int(rec["year"]),
                "starting_capital": round(start, 2),
                "realized_pnl": round(pnl, 2),
                "ending_capital": round(end, 2),
                "realized_return_pct": round(ret, 3),
            }
        )
        running = end

    return pd.DataFrame(rows, columns=cols)


def _validate_signal_data_quality(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate signal data quality per backtesting policy.
    
    Excludes rows with data gaps (missing price on signal_date).
    Flags rows with warm-up missing (price exists but lookback insufficient).
    
    Returns:
      (filtered_signals, audit_dict) where audit_dict has keys:
        - input_rows: total rows before validation
        - data_gap_excluded: missing price row for(ticker, signal_date)
        - valid_rows: rows passing validation
        - audit_details: list of {date, ticker, reason} for excluded rows
    """
    if signals_df.empty or prices_df.empty:
        return signals_df.copy(), {"input_rows": 0, "data_gap_excluded": 0, "valid_rows": 0, "audit_details": []}

    audit = {"input_rows": len(signals_df), "data_gap_excluded": 0, "audit_details": []}
    signals = signals_df.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
    prices = prices_df.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    # Normalize tickers and build price coverage map
    prices["Ticker"] = prices["Ticker"].astype(str).str.upper().str.strip()
    ticker_dates = {}
    for ticker_val in prices["Ticker"].unique():
        ticker_key = str(ticker_val).removesuffix(".NS")
        dates_set = set(prices[prices["Ticker"] == ticker_val]["Date"].dt.normalize())
        ticker_dates[ticker_key] = dates_set

    keep_mask = pd.Series(True, index=signals.index)
    for idx, row in signals.iterrows():
        sig_date = pd.to_datetime(row["signal_date"], errors="coerce")
        ticker_raw = str(row["ticker"]).upper().strip()
        ticker_key = ticker_raw.removesuffix(".NS")

        if pd.isna(sig_date):
            keep_mask.at[idx] = False
            audit["data_gap_excluded"] += 1
            audit["audit_details"].append({"date": str(row["signal_date"]), "ticker": ticker_raw, "reason": "invalid_date"})
            continue

        if ticker_key not in ticker_dates:
            keep_mask.at[idx] = False
            audit["data_gap_excluded"] += 1
            audit["audit_details"].append({"date": sig_date.strftime("%Y-%m-%d"), "ticker": ticker_raw, "reason": "no_price_history"})
            continue

        if sig_date.normalize() not in ticker_dates[ticker_key]:
            keep_mask.at[idx] = False
            audit["data_gap_excluded"] += 1
            audit["audit_details"].append({"date": sig_date.strftime("%Y-%m-%d"), "ticker": ticker_raw, "reason": "missing_price_on_date"})

    out = signals[keep_mask].copy()
    audit["data_gap_excluded"] = int(audit["data_gap_excluded"])
    audit["valid_rows"] = len(out)
    return out, audit


def build_oos_scoped_signals(
    signals_path: Path,
    prices_path: Path,
    *,
    predictions_path: Path,
    evaluation_mode: str,
    train_end_date: str,
    eval_hold_days: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base_signals = pd.read_csv(signals_path, parse_dates=["signal_date"])
    prices_df = pd.read_csv(prices_path, parse_dates=["Date"])
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path, parse_dates=["signal_date"])
        if "candidate_name" in predictions_df.columns:
            predictions_df = predictions_df[predictions_df["candidate_name"].astype(str).eq("scores_only")].copy()
        merge_keys = ["ticker", "signal_date", "pattern_family"]
        if "pattern" in base_signals.columns and "pattern" in predictions_df.columns:
            merge_keys.append("pattern")
        prediction_columns = merge_keys + [
            col for col in ["month", "signal_stop_risk", "signal_reliability_score", "stop_before_target"] if col in predictions_df.columns
        ]
        prediction_keys = predictions_df[prediction_columns].drop_duplicates(subset=merge_keys)
        scoped = base_signals.copy()
        scoped["signal_date"] = pd.to_datetime(scoped["signal_date"], errors="coerce")
        scoped = scoped.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()
        scoped = scoped.merge(prediction_keys, on=merge_keys, how="inner")
        summary = {
            "candidate_name": "scores_only",
            "evaluation_mode": evaluation_mode,
            "oos_rows": int(len(prediction_keys)),
        }
        return scoped, summary, prices_df

    signals_eval = _load_signals(signals_path, prices_df, max_hold_days=int(eval_hold_days))
    labels_df = _compute_labels(signals_eval, prices_df, target_pct=6.0, stop_pct=7.0, max_hold_days=int(eval_hold_days))
    parsed_train_end = pd.to_datetime(train_end_date, errors="coerce") if str(train_end_date).strip() else None
    summary, _, predictions_df = evaluate_candidate(
        "scores_only",
        CANDIDATE_SPECS["scores_only"],
        signals_eval,
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
        train_end_date=parsed_train_end,
        recency_half_life_months=3.0,
    )

    merge_keys = ["ticker", "signal_date", "pattern_family"]
    if "pattern" in base_signals.columns and "pattern" in predictions_df.columns:
        merge_keys.append("pattern")
    prediction_columns = merge_keys + [
        col for col in ["month", "signal_stop_risk", "signal_reliability_score", "stop_before_target"] if col in predictions_df.columns
    ]
    prediction_keys = predictions_df[prediction_columns].drop_duplicates(subset=merge_keys)
    scoped = base_signals.copy()
    scoped["signal_date"] = pd.to_datetime(scoped["signal_date"], errors="coerce")
    scoped = scoped.dropna(subset=["signal_date", "ticker", "pattern_family"]).copy()
    scoped = scoped.merge(prediction_keys, on=merge_keys, how="inner")
    return scoped, summary, prices_df


def filter_tracker_view(view: pd.DataFrame, *, max_days_held: int) -> pd.DataFrame:
    if view.empty:
        return view.copy()
    out = view.copy()
    out = out[pd.to_numeric(out["days_held"], errors="coerce").fillna(10**9) <= int(max_days_held)].copy()
    return out


def main() -> None:
    args = parse_args()
    scoped_signals, oos_summary, prices_df = build_oos_scoped_signals(
        Path(args.signals),
        Path(args.prices),
        predictions_path=Path(args.predictions),
        evaluation_mode=args.evaluation_mode,
        train_end_date=args.train_end_date,
        eval_hold_days=args.eval_hold_days,
    )

    # Validate data quality per backtesting policy
    scoped_signals, qa_audit = _validate_signal_data_quality(scoped_signals, prices_df)
    print(f"\nData Quality Validation:")
    print(f"  Input rows: {qa_audit['input_rows']}")
    print(f"  Excluded (data gap): {qa_audit['data_gap_excluded']}")
    print(f"  Valid rows: {qa_audit['valid_rows']}")
    if qa_audit['audit_details']:
        print(f"\n  Excluded details (sample of {min(5, len(qa_audit['audit_details']))}):")
        for item in qa_audit['audit_details'][:5]:
            print(f"    {item['date']} {item['ticker']}: {item['reason']}")

    scoped_signals = _apply_lab_stop_mode(
        scoped_signals,
        prices_df,
        stop_mode=args.stop_mode,
        fixed_stop_pct=float(args.stop_pct),
        atr_period=int(args.atr_period),
        atr_multiplier=float(args.atr_mult),
        structure_lookback=int(args.structure_lookback),
        structure_atr_buffer=float(args.atr_mult),
    )
    scoped_signals = _annotate_hold_to_target_only(scoped_signals, args.stop_mode)

    if float(args.min_score) > 0:
        scoped_signals = scoped_signals[pd.to_numeric(scoped_signals.get("signal_score"), errors="coerce").fillna(0.0) >= float(args.min_score)].copy()
    scoped_signals = filter_signals_by_catalyst_mode(scoped_signals, args.catalyst_mode)

    if str(args.capital_mode) == "reinvest_parallel":
        tracker = build_signal_tracker_reinvest_parallel(
            scoped_signals,
            prices_df,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            initial_capital=float(args.initial_capital),
        )
    else:
        tracker = build_signal_tracker(
            scoped_signals,
            prices_df,
            target_pct=float(args.target_pct),
            stop_pct=float(args.stop_pct),
            capital_per_trade=float(args.capital_per_trade),
        )
    tracker_view = filter_tracker_view(tracker, max_days_held=int(args.max_days_held))
    summary = summarize_signal_tracker(tracker_view)
    avg_trade_return_pct = float(pd.to_numeric(tracker_view.get("return_pct"), errors="coerce").mean()) if not tracker_view.empty else 0.0

    reinvest_enabled = bool("capital_mode" in tracker_view.columns and tracker_view["capital_mode"].astype(str).eq("reinvest_parallel").any())
    initial_capital = float(args.initial_capital) if reinvest_enabled else 0.0
    final_capital = float(initial_capital + float(summary.get("total_pnl", 0.0))) if reinvest_enabled else 0.0
    reinvest_return_pct = (((final_capital / initial_capital) - 1.0) * 100.0) if reinvest_enabled and initial_capital > 0 else 0.0

    summary_row = {
        "evaluation_mode": args.evaluation_mode,
        "train_end_date": args.train_end_date,
        "eval_hold_days": int(args.eval_hold_days),
        "stop_mode": args.stop_mode,
        "target_pct": float(args.target_pct),
        "stop_pct": float(args.stop_pct),
        "capital_mode": str(args.capital_mode),
        "initial_capital": float(args.initial_capital),
        "capital_per_trade": float(args.capital_per_trade),
        "min_score": float(args.min_score),
        "max_days_held": int(args.max_days_held),
        "catalyst_mode": args.catalyst_mode,
        "oos_rows": int(oos_summary.get("oos_rows", 0) or 0),
        "scoped_rows": int(len(scoped_signals)),
        "data_gap_excluded": int(qa_audit["data_gap_excluded"]),
        "valid_rows": int(qa_audit["valid_rows"]),
        "view_rows": int(len(tracker_view)),
        "avg_trade_return_pct": round(avg_trade_return_pct, 3),
        "final_capital": round(final_capital, 3),
        "total_profit": round(float(summary.get("total_pnl", 0.0)), 3),
        "reinvest_return_pct": round(reinvest_return_pct, 3),
        **{key: round(float(value), 3) if isinstance(value, float) else value for key, value in summary.items()},
    }
    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(Path(args.summary_out), index=False)
    tracker_view.to_csv(Path(args.view_out), index=False)

    yearly_df = summarize_reinvest_yearly(tracker_view)
    if str(args.yearly_out).strip() and not yearly_df.empty:
        yearly_df.to_csv(Path(args.yearly_out), index=False)
        print(f"Saved yearly summary to {args.yearly_out}")

    print(f"Saved summary to {args.summary_out}")
    print(f"Saved view to {args.view_out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()