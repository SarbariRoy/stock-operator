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

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SIGNALS = DATA_DIR / "signals_pattern_a.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pattern A triggers from OHLCV data")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Input prices CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_SIGNALS), help="Output signals CSV path")
    parser.add_argument("--as-of-date", type=str, default=None, help="Signal date YYYY-MM-DD (default: latest date)")
    parser.add_argument("--breakout-days", type=int, default=40, help="Breakout lookback window in trading days")
    parser.add_argument("--volume-multiplier", type=float, default=1.5, help="Volume spike threshold vs 20D average")
    parser.add_argument("--stop-pct", type=float, default=7.0, help="Initial stop loss percent below entry")
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
        stop_price = entry_price * (1.0 - stop_pct / 100.0)

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
                "stop_pct": stop_pct,
                "stop_price": round(stop_price, 4),
            }
        )

    if not all_rows:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    out = pd.DataFrame(all_rows)
    out.sort_values(["signal_date", "ticker"], inplace=True)
    return out


def main() -> None:
    args = parse_args()

    prices = load_prices(Path(args.prices))
    as_of_date = pd.to_datetime(args.as_of_date) if args.as_of_date else prices["Date"].max()

    signals = compute_signals(
        prices,
        as_of_date=as_of_date,
        breakout_days=args.breakout_days,
        volume_multiplier=args.volume_multiplier,
        stop_pct=args.stop_pct,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path, index=False)

    print(f"As-of date: {as_of_date.date().isoformat()}")
    print(f"Signals generated: {len(signals)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
