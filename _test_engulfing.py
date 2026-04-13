"""Quick regression checks for the bullish engulfing enhancer."""

from __future__ import annotations

import pandas as pd

from stock_triggers.ui.enhancers.bullish_engulfing import check, check_basic, check_confirmed


def _build_history(*, strong_context: bool) -> pd.DataFrame:
    rows: list[dict] = []
    start = pd.Timestamp("2025-01-01")

    if strong_context:
        base_prices = [90.0 + (0.10 * idx) for idx in range(210)]
        tail = [110.2, 110.0, 109.8, 109.6, 109.4, 109.2, 109.0, 108.8]
        sequence = base_prices + tail
        for idx, close in enumerate(sequence):
            rows.append(
                {
                    "Date": start + pd.Timedelta(days=idx),
                    "Ticker": "TEST.NS",
                    "Open": close - 0.2,
                    "High": close + 0.4,
                    "Low": close - 0.6,
                    "Close": close,
                    "Volume": 100.0,
                }
            )
        rows.append(
            {
                "Date": start + pd.Timedelta(days=len(rows)),
                "Ticker": "TEST.NS",
                "Open": 107.0,
                "High": 107.2,
                "Low": 105.4,
                "Close": 106.0,
                "Volume": 100.0,
            }
        )
        rows.append(
            {
                "Date": start + pd.Timedelta(days=len(rows)),
                "Ticker": "TEST.NS",
                "Open": 105.2,
                "High": 108.6,
                "Low": 105.0,
                "Close": 108.0,
                "Volume": 220.0,
            }
        )
    else:
        base_prices = [120.0 - (0.08 * idx) for idx in range(210)]
        tail = [104.0, 111.0, 103.5, 112.0, 103.0, 111.5, 102.5, 110.5]
        sequence = base_prices + tail
        for idx, close in enumerate(sequence):
            rows.append(
                {
                    "Date": start + pd.Timedelta(days=idx),
                    "Ticker": "TEST.NS",
                    "Open": close + 2.0,
                    "High": close + 4.5,
                    "Low": close - 4.5,
                    "Close": close,
                    "Volume": 120.0,
                }
            )
        rows.append(
            {
                "Date": start + pd.Timedelta(days=len(rows)),
                "Ticker": "TEST.NS",
                "Open": 108.0,
                "High": 108.8,
                "Low": 103.0,
                "Close": 105.0,
                "Volume": 120.0,
            }
        )
        rows.append(
            {
                "Date": start + pd.Timedelta(days=len(rows)),
                "Ticker": "TEST.NS",
                "Open": 101.0,
                "High": 109.5,
                "Low": 100.0,
                "Close": 108.5,
                "Volume": 110.0,
            }
        )

    return pd.DataFrame(rows)


weak = _build_history(strong_context=False)
assert check_basic(weak, "TEST.NS") is True
assert check(weak, "TEST.NS") is True
assert check_confirmed(weak, "TEST.NS") is False

strong = _build_history(strong_context=True)
assert check_basic(strong, "TEST.NS") is True
assert check(strong, "TEST.NS") is True
assert check_confirmed(strong, "TEST.NS") is True

print("engulfing tests passed")