"""Quick regression checks for piercing-line detectors."""

from __future__ import annotations

import pandas as pd

from stock_triggers.ui.enhancers import piercing_line, piercing_variant


def _frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(date_str),
                "Ticker": "TEST.NS",
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": 100.0,
            }
            for date_str, open_price, high_price, low_price, close_price in rows
        ]
    )


strict = _frame(
    [
        ("2025-01-01", 120.0, 121.0, 108.0, 110.0),
        ("2025-01-02", 107.0, 117.0, 106.5, 116.0),
    ]
)
assert piercing_line.check(strict, "TEST.NS") is True
assert piercing_variant.check(strict, "TEST.NS") is True

practical = _frame(
    [
        ("2025-01-01", 120.0, 121.0, 108.0, 110.0),
        ("2025-01-02", 109.5, 117.0, 109.0, 116.0),
    ]
)
assert piercing_line.check(practical, "TEST.NS") is False
assert piercing_variant.check(practical, "TEST.NS") is True

print("piercing tests passed")