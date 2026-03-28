"""Pattern D – RSI Oversold Bounce.

RSI(14) dips below 30 then rebounds back above 30 on the signal date, while
the stock is in an uptrend (SMA50 > SMA200).
"""

from __future__ import annotations

import pandas as pd

from . import STANDARD_SIGNAL_COLS
from .scoring import build_score_components


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI as a full Series (used for crossover detection)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def detect(
    prices: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    volume_multiplier: float = 1.0,
    stop_pct: float = 7.0,
    rsi_threshold: float = 30.0,
    compute_rsi_fn=None,
) -> pd.DataFrame:
    """Return a DataFrame of RSI oversold-bounce signals for *as_of_date*."""

    all_rows: list[dict] = []
    for ticker, g in prices.groupby("Ticker", sort=True):
        g = g.copy().sort_values("Date")

        g["SMA50"] = g["Close"].rolling(50).mean()
        g["SMA200"] = g["Close"].rolling(200).mean()
        g["VolAvg20"] = g["Volume"].rolling(20).mean()
        g["SwingLow10"] = g["Low"].shift(1).rolling(10).min()
        g["RSI"] = _rsi_series(g["Close"], period=14)
        g["RSI_prev"] = g["RSI"].shift(1)

        row = g[g["Date"] == as_of_date]
        if row.empty:
            continue
        r = row.iloc[0]

        needed = [r["SMA50"], r["SMA200"], r["VolAvg20"], r["RSI"], r["RSI_prev"], r["SwingLow10"]]
        if any(pd.isna(v) for v in needed):
            continue

        # Trend filter
        if not float(r["SMA50"]) > float(r["SMA200"]):
            continue

        # RSI bounce: was below threshold, now above
        if not (float(r["RSI_prev"]) < rsi_threshold <= float(r["RSI"])):
            continue

        # Mild volume filter
        if float(r["VolAvg20"]) > 0:
            vol_ratio = float(r["Volume"]) / float(r["VolAvg20"])
        else:
            continue
        if vol_ratio < float(volume_multiplier) * 0.6:
            continue

        entry_price = float(r["Close"])
        fixed_stop = entry_price * (1.0 - float(stop_pct) / 100.0)
        stop_price = max(fixed_stop, float(r["SwingLow10"]))
        stop_pct_eff = ((entry_price - stop_price) / entry_price) * 100.0

        trend_strength_pct = ((float(r["SMA50"]) / float(r["SMA200"])) - 1.0) * 100.0
        # Setup strength: how sharply RSI bounced
        rsi_bounce = float(r["RSI"]) - float(r["RSI_prev"])
        setup_strength_pct = rsi_bounce * 1.5  # scale for scoring

        rsi_value = float(r["RSI"])

        scores = build_score_components(
            trend_strength_pct=trend_strength_pct,
            setup_strength_pct=setup_strength_pct,
            volume_ratio=vol_ratio,
            stop_pct_eff=stop_pct_eff,
            rsi_value=rsi_value,
        )

        all_rows.append(
            {
                "signal_date": as_of_date.date().isoformat(),
                "ticker": ticker,
                "pattern": "D_rsi_bounce",
                "pattern_family": "D",
                "entry_price": round(entry_price, 4),
                "stop_pct": round(float(stop_pct_eff), 2),
                "stop_price": round(stop_price, 4),
                "score_trend": scores[0],
                "score_setup": scores[1],
                "score_volume": scores[2],
                "score_rsi": scores[4],
                "score_risk": scores[3],
                "signal_score": scores[5],
                "consensus_count": 1,
            }
        )

    if not all_rows:
        return pd.DataFrame(columns=STANDARD_SIGNAL_COLS)
    return pd.DataFrame(all_rows, columns=STANDARD_SIGNAL_COLS)
