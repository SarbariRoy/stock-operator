"""Stock selector: rank and allocate across stocks based on simple fundamentals and momentum.

This is intentionally similar in spirit to mutual-funds-selector.py, but works on a
single CSV of stock data that you maintain locally (no external APIs).

Expected input file (by default: data/stocks.csv):
- Ticker: stock symbol
- Name: company name (optional but recommended)
- Sector: sector / industry (optional)
- Price: latest price (float)
- MarketCap: market capitalization in crores or millions (float)
- AvgVolume: average daily trading volume (float)
- PE: trailing P/E ratio (float)
- PB: P/B ratio (float, optional)
- ROE: Return on Equity, % (float)
- Return_6M: 6-month price return, % (float)
- Return_12M: 12-month price return, % (float)
- Volatility_1Y: annualized volatility, % (float, optional)

You can adjust column names via CLI arguments if needed.

Scoring (higher composite_score is better):
- Momentum (weight 0.35): blend of 6M and 12M returns, higher is better.
- Profitability (weight 0.25): ROE, higher is better.
- Valuation (weight 0.20): P/E, lower is better (cheap vs peers).
- Volatility (weight 0.20): 1Y volatility, lower is better.

Basic filters (to avoid illiquid / tiny / penny stocks):
- Minimum price
- Minimum market cap
- Minimum average volume

The script prints a ranked table and an allocation of a user-specified budget
proportional to composite_score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _normalize(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    """Min-max normalize a numeric series to 0..1.

    If all values are (effectively) the same, return 1.0 everywhere so that
    this factor neither penalizes nor differentiates between candidates.
    """

    s = pd.to_numeric(series, errors="coerce").astype(float)
    if not higher_is_better:
        s = -s
    min_v, max_v = s.min(), s.max()
    if np.isclose(max_v, min_v) or np.isnan(min_v) or np.isnan(max_v):
        return pd.Series(1.0, index=series.index)
    return (s - min_v) / (max_v - min_v)


def load_stocks_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Stock data file not found: {path}. Create it with the columns described in stock-selector.py."
        )
    df = pd.read_csv(path)
    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    return df


def apply_basic_filters(
    df: pd.DataFrame,
    *,
    price_col: str,
    mcap_col: str,
    volume_col: str,
    min_price: float,
    min_mcap: float,
    min_avg_volume: float,
) -> pd.DataFrame:
    f = df.copy()
    f[price_col] = pd.to_numeric(f[price_col], errors="coerce")
    f[mcap_col] = pd.to_numeric(f[mcap_col], errors="coerce")
    f[volume_col] = pd.to_numeric(f[volume_col], errors="coerce")

    mask = (
        (f[price_col] >= min_price)
        & (f[mcap_col] >= min_mcap)
        & (f[volume_col] >= min_avg_volume)
    )
    return f[mask].reset_index(drop=True)


def build_scores(
    df: pd.DataFrame,
    *,
    pe_col: str,
    pb_col: Optional[str],
    roe_col: str,
    r6m_col: str,
    r12m_col: str,
    vol_col: Optional[str],
) -> pd.DataFrame:
    scored = df.copy()

    # Ensure numeric
    scored[pe_col] = pd.to_numeric(scored[pe_col], errors="coerce")
    if pb_col is not None and pb_col in scored.columns:
        scored[pb_col] = pd.to_numeric(scored[pb_col], errors="coerce")
    scored[roe_col] = pd.to_numeric(scored[roe_col], errors="coerce")
    scored[r6m_col] = pd.to_numeric(scored[r6m_col], errors="coerce")
    scored[r12m_col] = pd.to_numeric(scored[r12m_col], errors="coerce")
    if vol_col is not None and vol_col in scored.columns:
        scored[vol_col] = pd.to_numeric(scored[vol_col], errors="coerce")

    # Momentum: simple blend of 6M and 12M returns.
    scored["Momentum"] = 0.5 * scored[r6m_col] + 0.5 * scored[r12m_col]

    # Valuation: P/E (optionally adjusted by P/B if present).
    scored["PE_clean"] = scored[pe_col].replace({0.0: np.nan})
    if pb_col is not None and pb_col in scored.columns:
        scored["PB_clean"] = scored[pb_col].replace({0.0: np.nan})
    else:
        scored["PB_clean"] = np.nan

    # Profitability: ROE
    scored["ROE_clean"] = scored[roe_col]

    # Volatility: optional; if missing, treat all as equal.
    if vol_col is not None and vol_col in scored.columns:
        scored["Vol_clean"] = scored[vol_col]
    else:
        scored["Vol_clean"] = np.nan

    # Individual factor scores (0..1)
    scored["score_momentum"] = _normalize(scored["Momentum"], higher_is_better=True)
    scored["score_profit"] = _normalize(scored["ROE_clean"], higher_is_better=True)
    scored["score_valuation"] = _normalize(scored["PE_clean"], higher_is_better=False)
    scored["score_vol"] = _normalize(scored["Vol_clean"], higher_is_better=False)

    # Composite score – adjust weights to taste.
    scored["composite_score"] = (
        0.35 * scored["score_momentum"]
        + 0.25 * scored["score_profit"]
        + 0.20 * scored["score_valuation"]
        + 0.20 * scored["score_vol"]
    )

    return scored


def allocate(
    df: pd.DataFrame,
    *,
    budget: float,
    top_n: int,
    ticker_col: str,
    name_col: Optional[str],
) -> pd.DataFrame:
    ranked = df.sort_values("composite_score", ascending=False).head(top_n).copy()

    total_score = ranked["composite_score"].sum()
    if total_score <= 0:
        ranked["Allocation"] = 0.0
    else:
        ranked["Allocation"] = budget * ranked["composite_score"] / total_score

    # For an approximate expected 1Y return, use 12M return where available.
    if "Return_12M" in ranked.columns:
        expected_ret_pct = pd.to_numeric(ranked["Return_12M"], errors="coerce")
    else:
        expected_ret_pct = np.nan

    ranked["Expected_Return_%"] = expected_ret_pct
    ranked["Expected_Return_₹"] = ranked["Allocation"] * ranked["Expected_Return_%"] / 100.0

    cols = [ticker_col]
    if name_col and name_col in ranked.columns:
        cols.append(name_col)
    for extra in [
        "Sector",
        "Price",
        "MarketCap",
        "AvgVolume",
        "Momentum",
        "ROE_clean",
        "PE_clean",
        "PB_clean",
        "Vol_clean",
        "composite_score",
        "Allocation",
        "Expected_Return_%",
        "Expected_Return_₹",
    ]:
        if extra in ranked.columns and extra not in cols:
            cols.append(extra)

    return ranked[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank and allocate across stocks using a simple factor model.")

    parser.add_argument(
        "--file",
        type=str,
        default=str(DATA_DIR / "stocks.csv"),
        help="Path to CSV file with stock data.",
    )
    parser.add_argument("--budget", type=float, default=50_000.0, help="Budget to allocate across top stocks (₹).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top stocks to select.")

    # Column name overrides (in case your CSV uses different labels)
    parser.add_argument("--ticker-col", type=str, default="Ticker", help="Column name for ticker symbol.")
    parser.add_argument("--name-col", type=str, default="Name", help="Column name for company name (optional).")
    parser.add_argument("--price-col", type=str, default="Price", help="Column name for latest price.")
    parser.add_argument("--mcap-col", type=str, default="MarketCap", help="Column name for market cap.")
    parser.add_argument("--volume-col", type=str, default="AvgVolume", help="Column name for average volume.")
    parser.add_argument("--pe-col", type=str, default="PE", help="Column name for P/E ratio.")
    parser.add_argument("--pb-col", type=str, default="PB", help="Column name for P/B ratio (optional).")
    parser.add_argument("--roe-col", type=str, default="ROE", help="Column name for ROE in %.")
    parser.add_argument("--r6m-col", type=str, default="Return_6M", help="Column name for 6M return in %.")
    parser.add_argument("--r12m-col", type=str, default="Return_12M", help="Column name for 12M return in %.")
    parser.add_argument(
        "--vol-col",
        type=str,
        default="Volatility_1Y",
        help="Column name for 1Y volatility in % (optional).",
    )

    # Basic filters
    parser.add_argument("--min-price", type=float, default=50.0, help="Minimum stock price to consider.")
    parser.add_argument("--min-mcap", type=float, default=500.0, help="Minimum market cap to consider.")
    parser.add_argument("--min-avg-volume", type=float, default=50_000.0, help="Minimum average volume to consider.")

    args = parser.parse_args()

    csv_path = Path(args.file)
    raw = load_stocks_csv(csv_path)

    filtered = apply_basic_filters(
        raw,
        price_col=args.price_col,
        mcap_col=args.mcap_col,
        volume_col=args.volume_col,
        min_price=args.min_price,
        min_mcap=args.min_mcap,
        min_avg_volume=args.min_avg_volume,
    )

    if filtered.empty:
        raise SystemExit("No stocks left after applying basic filters. Relax your thresholds or check the input data.")

    scored = build_scores(
        filtered,
        pe_col=args.pe_col,
        pb_col=args.pb_col,
        roe_col=args.roe_col,
        r6m_col=args.r6m_col,
        r12m_col=args.r12m_col,
        vol_col=args.vol_col,
    )

    plan = allocate(
        scored,
        budget=args.budget,
        top_n=args.top_n,
        ticker_col=args.ticker_col,
        name_col=args.name_col,
    )

    print(f"Suggested allocation for ₹{args.budget:,.0f} (INR) across top {args.top_n} stocks:")
    formatters = {
        "Allocation": lambda x: f"₹{x:,.0f}",
        "Expected_Return_₹": lambda x: f"₹{x:,.0f}",
        "Expected_Return_%": lambda x: "" if pd.isna(x) else f"{x:.2f}%",
    }
    # Use to_string for a clean, wide table.
    print(plan.to_string(index=False, formatters={k: v for k, v in formatters.items() if k in plan.columns}))


if __name__ == "__main__":
    main()
