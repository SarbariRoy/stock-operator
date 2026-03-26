"""Generate stock health scores and metrics from prices_eod.csv.

Computes trend, momentum, and price strength metrics for each stock
and saves to stock_scores.csv for use in dashboards and filters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "stock_triggers" / "data"
DEFAULT_PRICES = DATA_DIR / "prices_eod.csv"
DEFAULT_SCORES = DATA_DIR / "stock_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stock health scores from daily prices")
    parser.add_argument("--prices", type=str, default=str(DEFAULT_PRICES), help="Input prices CSV path")
    parser.add_argument("--out", type=str, default=str(DEFAULT_SCORES), help="Output scores CSV path")
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


def pct_return_from_offset(series: pd.Series, offset: int) -> float | None:
    """Compute % return from offset row in series."""
    if series.empty or len(series) <= offset:
        return None
    latest = float(series.iloc[-1])
    old = float(series.iloc[-1 - offset])
    if old == 0:
        return None
    return ((latest / old) - 1.0) * 100.0


def compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    """Compute classic Wilder's RSI for the provided close-price series.

    Returns the latest RSI value (0-100) or None if there isn't enough history.
    """

    if series is None or len(series) < period + 1:
        return None

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: use exponential weighted mean with com = period-1
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])

    if last_loss == 0:
        # No losses in lookback window: treat as overbought (RSI=100)
        return 100.0

    rs = last_gain / last_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(round(rsi, 2))


def compute_stock_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute health score, trend metrics, and insights for each stock."""
    rows: list[dict] = []

    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")
        close = g["Close"].astype(float)

        sma20 = close.rolling(20).mean().iloc[-1] if len(g) >= 20 else None
        sma50 = close.rolling(50).mean().iloc[-1] if len(g) >= 50 else None
        sma200 = close.rolling(200).mean().iloc[-1] if len(g) >= 200 else None

        latest_date = pd.to_datetime(g["Date"].iloc[-1]).date().isoformat()
        latest_close = float(close.iloc[-1])
        high_52w = float(g["High"].tail(252).max()) if "High" in g.columns else float(close.tail(252).max())
        low_52w = float(g["Low"].tail(252).min()) if "Low" in g.columns else float(close.tail(252).min())
        dist_high_pct = ((latest_close / high_52w) - 1.0) * 100.0 if high_52w else None

        ret_1d = pct_return_from_offset(close, 1)
        ret_5d = pct_return_from_offset(close, 5)
        ret_20d = pct_return_from_offset(close, 20)
        ret_60d = pct_return_from_offset(close, 60)
        rsi14 = compute_rsi(close, period=14)

        # Compute health score: 0-4 points
        score = 0
        if sma50 is not None and sma200 is not None and sma50 > sma200:
            score += 1
        if ret_20d is not None and ret_20d > 0:
            score += 1
        if ret_60d is not None and ret_60d > 0:
            score += 1
        if dist_high_pct is not None and dist_high_pct >= -12:
            score += 1

        if score >= 3:
            health = "Doing well"
            insight = "Trend is strong and price behavior is healthy. Keep on watchlist for future opportunities."
        elif score == 2:
            health = "Mixed"
            insight = "Signals are mixed. Wait for trend and momentum to align before fresh allocation."
        else:
            health = "Weak"
            insight = "Trend is weak right now. Better to avoid fresh long entries until structure improves."

        rows.append(
            {
                "ticker": ticker,
                "latest_date": latest_date,
                "latest_close": round(latest_close, 2),
                "ret_1d_pct": round(ret_1d, 2) if ret_1d is not None else None,
                "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
                "ret_20d_pct": round(ret_20d, 2) if ret_20d is not None else None,
                "ret_60d_pct": round(ret_60d, 2) if ret_60d is not None else None,
                "rsi14": rsi14,
                "sma20": round(float(sma20), 2) if sma20 is not None and pd.notna(sma20) else None,
                "sma50": round(float(sma50), 2) if sma50 is not None and pd.notna(sma50) else None,
                "sma200": round(float(sma200), 2) if sma200 is not None and pd.notna(sma200) else None,
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "dist_from_52w_high_pct": round(dist_high_pct, 2) if dist_high_pct is not None else None,
                "health": health,
                "score": score,
                "insight": insight,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.sort_values(["score", "ret_20d_pct", "ret_60d_pct"], ascending=False, inplace=True)
    return out


def main() -> None:
    args = parse_args()
    prices = load_prices(Path(args.prices))
    scores_df = compute_stock_scores(prices)

    if scores_df.empty:
        print("Warning: No stock scores computed.")
    else:
        out_path = Path(args.out)
        scores_df.to_csv(out_path, index=False)
        print(f"Stock scores written to {out_path} ({len(scores_df)} stocks)")


if __name__ == "__main__":
    main()
