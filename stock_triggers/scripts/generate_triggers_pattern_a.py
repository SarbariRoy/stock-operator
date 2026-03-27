"""Generate Pattern A breakout triggers from prices_eod.csv.

Pattern A conditions (as-of date):
- SMA50 > SMA200
- Close > SMA50 and Close > SMA200
- Close > previous N-day highest close (default N=40)
- Volume >= volume_multiplier * 20-day average volume (default 1.5x)

Output is written to stock_triggers/data/signals_pattern_a.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_stock_scores import compute_rsi

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_pattern_a.csv"
DEFAULT_SELL_SIGNALS = DATA_DIR / "sell_signals_pattern_a.csv"

# Component weights for Pattern A scoring
WEIGHT_TREND = 0.28
WEIGHT_SETUP = 0.28
WEIGHT_VOLUME = 0.19
WEIGHT_RISK = 0.20
WEIGHT_RSI = 0.05


def _clip_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _build_score_components(
    *,
    trend_strength_pct: float,
    setup_strength_pct: float,
    volume_ratio: float,
    stop_pct_eff: float,
    rsi_value: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Build per-signal component scores and a combined signal_score.

    This mirrors the scoring logic used in the Streamlit UI so that
    signals_pattern_a.csv carries the same signal_score semantics.
    """

    score_trend = _clip_score(50.0 + trend_strength_pct * 5.0)
    score_setup = _clip_score(50.0 + setup_strength_pct * 8.0)
    score_volume = _clip_score(40.0 + volume_ratio * 20.0)
    score_risk = _clip_score(100.0 - stop_pct_eff * 6.0)

    # RSI component: map latest RSI value (0-100) into a 0-100 score.
    # If RSI is missing, treat it as neutral (50).
    if rsi_value is None or pd.isna(rsi_value):
        score_rsi = 50.0
    else:
        score_rsi = _clip_score(rsi_value)

    signal_score = round(
        (WEIGHT_TREND * score_trend)
        + (WEIGHT_SETUP * score_setup)
        + (WEIGHT_VOLUME * score_volume)
        + (WEIGHT_RISK * score_risk)
        + (WEIGHT_RSI * score_rsi),
        1,
    )
    return (
        round(score_trend, 1),
        round(score_setup, 1),
        round(score_volume, 1),
        round(score_risk, 1),
        round(score_rsi, 1),
        signal_score,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pattern A triggers from OHLCV data")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Input prices CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNALS), help="Output signals CSV path")
    parser.add_argument(
        "--sell-out",
        type=str,
        default=str(DEFAULT_SELL_SIGNALS),
        help="Output sell signals CSV path",
    )
    parser.add_argument("--as-of-date", type=str, default=None, help="Signal date YYYY-MM-DD (default: latest date)")
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Generate buy triggers for all available dates in prices file (appended with de-dup).",
    )
    parser.add_argument("--breakout-days", type=int, default=40, help="Breakout lookback window in trading days")
    parser.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume spike threshold vs 20D average")
    parser.add_argument("--stop-pct", type=float, default=7.0, help="Initial stop loss percent below entry")
    parser.add_argument(
        "--target-return-pct",
        type=float,
        default=6.0,
        help="Return %% target that emits a sell trigger (default: 6.0)",
    )
    return parser.parse_args()


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


def compute_signals(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
) -> pd.DataFrame:
    all_rows: list[dict] = []

    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["PrevNHighClose"] = g["Close"].shift(1).rolling(breakout_days).max()

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["PrevNHighClose"]]
        if any(pd.isna(v) for v in needed):
            continue

        cond_trend = bool(r["SMA50"] > r["SMA200"])
        cond_price = bool((r["Close"] > r["SMA50"]) and (r["Close"] > r["SMA200"]))
        cond_breakout = bool(r["Close"] > r["PrevNHighClose"])
        cond_volume = bool(r["Volume"] >= volume_multiplier * r["VolAvg20"])

        if not (cond_trend and cond_price and cond_breakout and cond_volume):
            continue

        entry_price = float(r["Close"])
        stop_price = entry_price * (1.0 - float(stop_pct) / 100.0)

        # Compute scoring inputs using the same recipe as the UI backtest code.
        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        setup_strength_pct = ((float(r["Close"]) / float(r["PrevNHighClose"])) - 1.0) * 100.0
        volume_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        stop_pct_eff = (entry_price - stop_price) / entry_price * 100.0

        # RSI at signal date based on closing prices up to as_of_date.
        hist_close = g[g["Date"] <= as_of_date]["Close"].astype(float)
        rsi_value = compute_rsi(hist_close, period=14)

        score_trend, score_setup, score_volume, score_risk, score_rsi, signal_score = _build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=volume_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": f"A_breakout_{breakout_days}d",
                "close": round(entry_price, 4),
                "sma50": round(float(r["SMA50"]), 4),
                "sma200": round(float(r["SMA200"]), 4),
                "prev_high_close": round(float(r["PrevNHighClose"]), 4),
                "volume": int(r["Volume"]),
                "vol_avg20": round(float(r["VolAvg20"]), 2),
                "entry_price": round(entry_price, 4),
                "entry_band_low": round(entry_price, 4),
                "entry_band_high": round(entry_price * 1.02, 4),
                "stop_pct": float(stop_pct),
                "stop_price": round(stop_price, 4),
                "pattern_family": "A",
                "score_trend": score_trend,
                "score_setup": score_setup,
                "score_volume": score_volume,
                "score_risk": score_risk,
                "score_rsi": score_rsi,
                "signal_score": signal_score,
                "consensus_count": 1,
            }
        )

    if not all_rows:
        return pd.DataFrame(columns=_buy_signal_columns())

    out = pd.DataFrame(all_rows)
    out.sort_values(["signal_date", "ticker"], inplace=True)
    return out


def compute_signals_for_all_dates(
    prices: pd.DataFrame,
    *,
    breakout_days: int,
    volume_multiplier: float,
    stop_pct: float,
) -> pd.DataFrame:
    all_dates = sorted(prices["Date"].drop_duplicates())
    chunks: list[pd.DataFrame] = []

    for d in all_dates:
        day = compute_signals(
            prices,
            as_of_date=d,
            breakout_days=breakout_days,
            volume_multiplier=volume_multiplier,
            stop_pct=stop_pct,
        )
        if not day.empty:
            chunks.append(day)

    if not chunks:
        return pd.DataFrame(columns=_buy_signal_columns())

    out = pd.concat(chunks, ignore_index=True)
    out.drop_duplicates(subset=["signal_date", "ticker", "pattern"], keep="last", inplace=True)
    out.sort_values(["signal_date", "ticker", "pattern"], inplace=True)
    return out


def _buy_signal_columns() -> list[str]:
    return [
        "signal_date",
        "ticker",
        "pattern",
        "close",
        "sma50",
        "sma200",
        "prev_high_close",
        "volume",
        "vol_avg20",
        "entry_price",
        "entry_band_low",
        "entry_band_high",
        "stop_pct",
        "stop_price",
        "pattern_family",
        "score_trend",
        "score_setup",
        "score_volume",
        "score_rsi",
        "score_risk",
        "signal_score",
        "consensus_count",
    ]


def merge_buy_signals(existing_path: Path, new_signals: pd.DataFrame) -> pd.DataFrame:
    cols = _buy_signal_columns()
    existing = pd.DataFrame(columns=cols)
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        for c in cols:
            if c not in existing.columns:
                existing[c] = pd.NA
        existing = existing[cols]

    if existing.empty:
        merged = new_signals[cols].copy()
    elif new_signals.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_signals[cols]], ignore_index=True)
    merged.drop_duplicates(subset=["signal_date", "ticker", "pattern"], keep="last", inplace=True)
    merged.sort_values(["signal_date", "ticker", "pattern"], inplace=True)
    return merged


def compute_sell_signals(
    buy_signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    target_return_pct: float,
) -> pd.DataFrame:
    sell_rows: list[dict] = []

    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])

    for _, sig in buy_signals.iterrows():
        ticker = str(sig["ticker"])
        pattern = str(sig["pattern"])
        buy_date = pd.to_datetime(sig["signal_date"])
        entry_price = float(sig["entry_price"])

        target_price = entry_price * (1.0 + target_return_pct / 100.0)

        future = prices[(prices["Ticker"] == ticker) & (prices["Date"] > buy_date)].copy()
        if future.empty:
            continue

        hit = future[future["Close"] >= target_price]
        if hit.empty:
            continue

        first_hit = hit.sort_values("Date").iloc[0]
        sell_price = float(first_hit["Close"])
        sell_date = pd.to_datetime(first_hit["Date"]).date().isoformat()
        return_pct = ((sell_price - entry_price) / entry_price) * 100.0

        sell_rows.append(
            {
                "buy_signal_date": buy_date.date().isoformat(),
                "sell_signal_date": sell_date,
                "ticker": ticker,
                "pattern": pattern,
                "entry_price": round(entry_price, 4),
                "target_return_pct": float(target_return_pct),
                "target_price": round(target_price, 4),
                "sell_price": round(sell_price, 4),
                "realized_return_pct": round(return_pct, 2),
            }
        )

    cols = [
        "buy_signal_date",
        "sell_signal_date",
        "ticker",
        "pattern",
        "entry_price",
        "target_return_pct",
        "target_price",
        "sell_price",
        "realized_return_pct",
    ]
    if not sell_rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(sell_rows, columns=cols)
    out.drop_duplicates(subset=["buy_signal_date", "ticker", "pattern"], keep="first", inplace=True)
    out.sort_values(["sell_signal_date", "ticker", "pattern"], inplace=True)
    return out


def merge_sell_signals(existing_path: Path, new_sell_signals: pd.DataFrame) -> pd.DataFrame:
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame(columns=list(new_sell_signals.columns))

    if existing.empty:
        merged = new_sell_signals.copy()
    elif new_sell_signals.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new_sell_signals], ignore_index=True)
    if not merged.empty:
        merged.drop_duplicates(subset=["buy_signal_date", "ticker", "pattern"], keep="first", inplace=True)
        merged.sort_values(["sell_signal_date", "ticker", "pattern"], inplace=True)
    return merged


def main() -> None:
    args = parse_args()

    prices = load_prices(Path(args.prices))
    as_of_date = pd.to_datetime(args.as_of_date) if args.as_of_date else prices["Date"].max()

    if args.backfill_history:
        new_signals = compute_signals_for_all_dates(
            prices,
            breakout_days=args.breakout_days,
            volume_multiplier=args.volume_multiplier,
            stop_pct=args.stop_pct,
        )
    else:
        new_signals = compute_signals(
            prices,
            as_of_date=as_of_date,
            breakout_days=args.breakout_days,
            volume_multiplier=args.volume_multiplier,
            stop_pct=args.stop_pct,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_buy_signals = merge_buy_signals(out_path, new_signals)
    all_buy_signals.to_csv(out_path, index=False)

    sell_out_path = Path(args.sell_out)
    sell_out_path.parent.mkdir(parents=True, exist_ok=True)
    new_sell_signals = compute_sell_signals(
        all_buy_signals,
        prices,
        target_return_pct=args.target_return_pct,
    )
    all_sell_signals = merge_sell_signals(sell_out_path, new_sell_signals)
    all_sell_signals.to_csv(sell_out_path, index=False)

    if args.backfill_history:
        print("As-of date: backfill all available dates")
        print(f"New buy signals generated in backfill: {len(new_signals)}")
    else:
        print(f"As-of date: {as_of_date.date().isoformat()}")
        print(f"New buy signals generated today: {len(new_signals)}")
    print(f"Total buy signals tracked: {len(all_buy_signals)}")
    print(f"Total sell signals tracked (target {args.target_return_pct:.1f}%): {len(all_sell_signals)}")
    print(f"Buy signals saved to: {out_path}")
    print(f"Sell signals saved to: {sell_out_path}")


if __name__ == "__main__":
    main()
