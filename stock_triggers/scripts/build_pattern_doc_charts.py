from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRICES_PATH = ROOT / "stock_triggers" / "data" / "prices_eod.csv"
SIGNALS_PATH = ROOT / "stock_triggers" / "data" / "signals_all_patterns.csv"
OUT_DIR = ROOT / "stock_triggers" / "docs" / "assets" / "pattern-charts"

WIDTH = 1400
HEIGHT = 980
MARGIN_LEFT = 72
MARGIN_RIGHT = 28
MARGIN_TOP = 68
MARGIN_BOTTOM = 46
GAP = 22
PRICE_H = 430
VOL_H = 125
RSI_H = 125
MACD_H = 125


@dataclass(frozen=True)
class ChartExample:
    family: str
    ticker: str
    signal_date: str
    pattern: str
    output_name: str
    title: str


EXAMPLES: list[ChartExample] = [
    ChartExample("A", "ADANIPOWER.NS", "2025-09-19", "A_breakout_40d", "pattern-a-breakout.svg", "Pattern A: Trend Breakout With Volume"),
    ChartExample("B", "BHARTIARTL.NS", "2025-12-30", "B_pullback_rebound", "pattern-b-pullback.svg", "Pattern B: Pullback And Rebound Near SMA20"),
    ChartExample("C", "ONGC.NS", "2026-03-27", "C_macd_crossover", "pattern-c-macd.svg", "Pattern C: MACD Bullish Crossover"),
    ChartExample("D", "PIDILITIND.NS", "2025-08-06", "D_rsi_bounce", "pattern-d-rsi.svg", "Pattern D: RSI Oversold Bounce"),
    ChartExample("E", "BEL.NS", "2025-06-20", "E_boll_squeeze", "pattern-e-squeeze.svg", "Pattern E: Bollinger Squeeze Breakout"),
    ChartExample("F", "COALINDIA.NS", "2026-03-04", "F_vwap_reclaim", "pattern-f-vwap.svg", "Pattern F: VWAP Reclaim"),
    ChartExample("G", "BRITANNIA.NS", "2024-09-12", "G_vcp_breakout", "pattern-g-vcp.svg", "Pattern G: VCP Breakout"),
]

COLORS = {
    "bg": "#fbfaf6",
    "grid": "#d9d5ca",
    "text": "#242424",
    "muted": "#66605a",
    "up": "#1b8a5a",
    "down": "#d14b5a",
    "sma20": "#ad7c1c",
    "sma50": "#3f6db5",
    "sma200": "#7b4fb3",
    "rsi": "#7b4fb3",
    "macd": "#2b6cb0",
    "signal": "#cc5a71",
    "highlight": "#c97b18",
    "vwap": "#2962cc",
    "bb": "#9fbc8b",
}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("Date")
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    out["VolAvg20"] = out["Volume"].rolling(20).mean()
    out["RSI14"] = compute_rsi(out["Close"], period=14)
    out["EMA12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    out["BB_MID"] = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_UPPER"] = out["BB_MID"] + (2.0 * bb_std)
    out["BB_LOWER"] = out["BB_MID"] - (2.0 * bb_std)
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    out["VWAP20"] = (tp * out["Volume"]).rolling(20).sum() / out["Volume"].rolling(20).sum()
    return out


def breakout_days_from_pattern(pattern: str) -> int:
    match = re.search(r"(\d+)d", str(pattern or ""))
    return int(match.group(1)) if match else 40


def fmt_num(value: float | int | str | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def make_scale(values: list[float], top: float, height: float):
    low = min(values)
    high = max(values)
    pad = (high - low) * 0.08 if high > low else max(abs(high) * 0.05, 1.0)
    low -= pad
    high += pad

    def scale(value: float) -> float:
        if high == low:
            return top + (height / 2.0)
        return top + height - ((value - low) / (high - low)) * height

    return scale


def x_positions(count: int) -> list[float]:
    usable = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    if count == 1:
        return [MARGIN_LEFT + usable / 2.0]
    step = usable / (count - 1)
    return [MARGIN_LEFT + idx * step for idx in range(count)]


def svg_line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1.0, dash: str | None = None, opacity: float | None = None) -> str:
    attrs = [f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"', f'stroke="{stroke}"', f'stroke-width="{width:.2f}"']
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<line {' '.join(attrs)}/>"


def svg_rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str | None = None, stroke_width: float = 0.0, opacity: float | None = None, rx: float | None = None) -> str:
    attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'width="{width:.2f}"', f'height="{height:.2f}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width:.2f}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    if rx is not None:
        attrs.append(f'rx="{rx:.2f}"')
    return f"<rect {' '.join(attrs)}/>"


def svg_text(x: float, y: float, text: str, *, size: int = 13, weight: str = "400", fill: str | None = None, anchor: str = "start") -> str:
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def svg_path(points: list[tuple[float, float]], *, stroke: str, width: float = 1.5, fill: str = "none", opacity: float | None = None) -> str:
    if not points:
        return ""
    d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
    attrs = [f'd="{d}"', f'stroke="{stroke}"', f'stroke-width="{width:.2f}"', f'fill="{fill}"']
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<path {' '.join(attrs)}/>"


def segmented_paths(xs: list[float], series: pd.Series, scaler) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, value in zip(xs, series, strict=False):
        if pd.notna(value):
            current.append((x, scaler(float(value))))
        elif current:
            if len(current) >= 2:
                segments.append(current)
            current = []
    if len(current) >= 2:
        segments.append(current)
    return segments


def draw_legend(items: list[tuple[str, str]], x: float, y: float) -> list[str]:
    parts: list[str] = []
    cursor = x
    for color, label in items:
        parts.append(svg_line(cursor, y - 5, cursor + 18, y - 5, stroke=color, width=2.6))
        parts.append(svg_text(cursor + 24, y, label, size=11, fill=COLORS["muted"]))
        cursor += 24 + len(label) * 7.0 + 20
    return parts


def build_chart(example: ChartExample, prices: pd.DataFrame, signals: pd.DataFrame) -> str:
    signal_date = pd.Timestamp(example.signal_date)
    matches = signals[
        (signals["pattern_family"].astype(str) == example.family)
        & (signals["ticker"].astype(str) == example.ticker)
        & (signals["pattern"].astype(str) == example.pattern)
        & (signals["signal_date"] == signal_date)
    ]
    if matches.empty:
        raise ValueError(f"No signal row found for {example}")
    signal_row = matches.sort_values("signal_score", ascending=False).iloc[0]

    ticker_prices = prices[prices["Ticker"].astype(str) == example.ticker].copy().sort_values("Date")
    ticker_prices = add_indicators(ticker_prices)
    if signal_date not in set(ticker_prices["Date"]):
        raise ValueError(f"No price row found for {example.ticker} on {example.signal_date}")

    signal_loc = ticker_prices.index[ticker_prices["Date"] == signal_date][0]
    idx_pos = ticker_prices.index.get_loc(signal_loc)
    start = max(0, idx_pos - 70)
    end = min(len(ticker_prices), idx_pos + 11)
    frame = ticker_prices.iloc[start:end].reset_index(drop=True)
    signal_idx = int(frame.index[frame["Date"] == signal_date][0])
    xs = x_positions(len(frame))
    candle_width = max(5.0, min(11.0, ((WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / max(len(frame), 2)) * 0.65))

    price_top = MARGIN_TOP
    vol_top = price_top + PRICE_H + GAP
    rsi_top = vol_top + VOL_H + GAP
    macd_top = rsi_top + RSI_H + GAP

    price_values = frame[["Low", "High", "SMA20", "SMA50", "SMA200", "BB_UPPER", "BB_LOWER", "VWAP20"]].stack().dropna().astype(float).tolist()
    vol_values = [0.0] + frame[["Volume", "VolAvg20"]].stack().dropna().astype(float).tolist()
    rsi_values = [0.0, 30.0, 70.0, 100.0] + frame["RSI14"].dropna().astype(float).tolist()
    macd_values = [0.0] + frame[["MACD", "MACD_SIGNAL", "MACD_HIST"]].stack().dropna().astype(float).tolist()
    price_y = make_scale(price_values, price_top, PRICE_H)
    vol_y = make_scale(vol_values, vol_top, VOL_H)
    rsi_y = make_scale(rsi_values, rsi_top, RSI_H)
    macd_y = make_scale(macd_values, macd_top, MACD_H)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        svg_rect(0, 0, WIDTH, HEIGHT, fill=COLORS["bg"]),
        svg_text(MARGIN_LEFT, 34, example.title, size=26, weight="700"),
    ]

    subtitle = f"{example.ticker} | {example.signal_date} | pattern {example.pattern} | signal score {fmt_num(signal_row.get('signal_score'))}"
    parts.append(svg_text(MARGIN_LEFT, 57, subtitle, size=14, fill=COLORS["muted"]))

    for y in np.linspace(price_top, price_top + PRICE_H, 5):
        parts.append(svg_line(MARGIN_LEFT, float(y), WIDTH - MARGIN_RIGHT, float(y), stroke=COLORS["grid"], width=0.9, opacity=0.65))
    for y in np.linspace(vol_top, vol_top + VOL_H, 3):
        parts.append(svg_line(MARGIN_LEFT, float(y), WIDTH - MARGIN_RIGHT, float(y), stroke=COLORS["grid"], width=0.8, opacity=0.55))
    for y in np.linspace(rsi_top, rsi_top + RSI_H, 3):
        parts.append(svg_line(MARGIN_LEFT, float(y), WIDTH - MARGIN_RIGHT, float(y), stroke=COLORS["grid"], width=0.8, opacity=0.55))
    for y in np.linspace(macd_top, macd_top + MACD_H, 3):
        parts.append(svg_line(MARGIN_LEFT, float(y), WIDTH - MARGIN_RIGHT, float(y), stroke=COLORS["grid"], width=0.8, opacity=0.55))

    signal_x = xs[signal_idx]
    for top, height in [(price_top, PRICE_H), (vol_top, VOL_H), (rsi_top, RSI_H), (macd_top, MACD_H)]:
        parts.append(svg_line(signal_x, top, signal_x, top + height, stroke="#444", width=1.1, dash="6 5", opacity=0.75))

    for x, row in zip(xs, frame.itertuples(index=False), strict=False):
        candle_color = COLORS["up"] if float(row.Close) >= float(row.Open) else COLORS["down"]
        parts.append(svg_line(x, price_y(float(row.Low)), x, price_y(float(row.High)), stroke=candle_color, width=1.2))
        body_top = price_y(max(float(row.Open), float(row.Close)))
        body_bottom = price_y(min(float(row.Open), float(row.Close)))
        body_height = max(body_bottom - body_top, 2.2)
        parts.append(svg_rect(x - candle_width / 2, body_top, candle_width, body_height, fill=candle_color, stroke=candle_color, stroke_width=1.0, rx=1.0))

    price_series: list[tuple[pd.Series, str, str]] = [
        (frame["SMA20"], COLORS["sma20"], "SMA20"),
        (frame["SMA50"], COLORS["sma50"], "SMA50"),
        (frame["SMA200"], COLORS["sma200"], "SMA200"),
    ]
    if example.family == "E":
        price_series.extend([
            (frame["BB_UPPER"], COLORS["bb"], "BB Upper"),
            (frame["BB_LOWER"], COLORS["bb"], "BB Lower"),
        ])
    if example.family == "F":
        price_series.append((frame["VWAP20"], COLORS["vwap"], "VWAP20"))
    if example.family == "A":
        prev_days = breakout_days_from_pattern(example.pattern)
        price_series.append((frame["Close"].shift(1).rolling(prev_days).max(), COLORS["highlight"], f"Prev {prev_days}D High Close"))
    if example.family == "G":
        price_series.append((frame["High"].shift(1).rolling(20).max(), COLORS["highlight"], "Resistance"))

    legend_items: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for series, color, label in price_series:
        for segment in segmented_paths(xs, series, price_y):
            parts.append(svg_path(segment, stroke=color, width=1.6 if label != "SMA200" else 1.4, opacity=0.9))
        if label not in seen_labels:
            legend_items.append((color, label))
            seen_labels.add(label)

    for x, row in zip(xs, frame.itertuples(index=False), strict=False):
        bar_color = COLORS["up"] if float(row.Close) >= float(row.Open) else COLORS["down"]
        bar_top = vol_y(float(row.Volume))
        parts.append(svg_rect(x - candle_width / 2, bar_top, candle_width, (vol_top + VOL_H) - bar_top, fill=bar_color, opacity=0.55))
    for segment in segmented_paths(xs, frame["VolAvg20"], vol_y):
        parts.append(svg_path(segment, stroke=COLORS["highlight"], width=1.6))

    for segment in segmented_paths(xs, frame["RSI14"], rsi_y):
        parts.append(svg_path(segment, stroke=COLORS["rsi"], width=1.8))
    parts.append(svg_line(MARGIN_LEFT, rsi_y(70.0), WIDTH - MARGIN_RIGHT, rsi_y(70.0), stroke=COLORS["down"], width=1.0, dash="6 5", opacity=0.8))
    parts.append(svg_line(MARGIN_LEFT, rsi_y(30.0), WIDTH - MARGIN_RIGHT, rsi_y(30.0), stroke=COLORS["up"], width=1.0, dash="6 5", opacity=0.8))
    if example.family == "D":
        parts.append(svg_line(MARGIN_LEFT, rsi_y(35.0), WIDTH - MARGIN_RIGHT, rsi_y(35.0), stroke=COLORS["highlight"], width=1.0, dash="3 4", opacity=0.8))

    parts.append(svg_line(MARGIN_LEFT, macd_y(0.0), WIDTH - MARGIN_RIGHT, macd_y(0.0), stroke=COLORS["muted"], width=1.0, opacity=0.7))
    for x, value in zip(xs, frame["MACD_HIST"].fillna(0.0), strict=False):
        top_val = max(float(value), 0.0)
        bottom_val = min(float(value), 0.0)
        top_y = macd_y(top_val)
        bottom_y = macd_y(bottom_val)
        bar_color = COLORS["up"] if float(value) >= 0 else COLORS["down"]
        parts.append(svg_rect(x - candle_width / 2, top_y, candle_width, max(bottom_y - top_y, 1.6), fill=bar_color, opacity=0.45))
    for segment in segmented_paths(xs, frame["MACD"], macd_y):
        parts.append(svg_path(segment, stroke=COLORS["macd"], width=1.7))
    for segment in segmented_paths(xs, frame["MACD_SIGNAL"], macd_y):
        parts.append(svg_path(segment, stroke=COLORS["signal"], width=1.5))

    parts.extend(draw_legend(legend_items, MARGIN_LEFT, price_top + 18))
    parts.extend(draw_legend([(COLORS["highlight"], "VolAvg20")], MARGIN_LEFT, vol_top + 16))
    parts.extend(draw_legend([(COLORS["rsi"], "RSI14")], MARGIN_LEFT, rsi_top + 16))
    parts.extend(draw_legend([(COLORS["macd"], "MACD"), (COLORS["signal"], "Signal")], MARGIN_LEFT, macd_top + 16))

    parts.append(svg_text(16, price_top + 18, "Price", size=12, fill=COLORS["muted"]))
    parts.append(svg_text(16, vol_top + 18, "Volume", size=12, fill=COLORS["muted"]))
    parts.append(svg_text(16, rsi_top + 18, "RSI", size=12, fill=COLORS["muted"]))
    parts.append(svg_text(16, macd_top + 18, "MACD", size=12, fill=COLORS["muted"]))

    tick_step = max(len(frame) // 6, 1)
    tick_positions = list(range(0, len(frame), tick_step))
    if tick_positions[-1] != len(frame) - 1:
        tick_positions.append(len(frame) - 1)
    for pos in tick_positions:
        x = xs[pos]
        label = pd.Timestamp(frame.iloc[pos]["Date"]).strftime("%Y-%m-%d")
        parts.append(svg_line(x, macd_top + MACD_H, x, macd_top + MACD_H + 6, stroke=COLORS["muted"], width=0.9))
        parts.append(svg_text(x, HEIGHT - 12, label, size=11, fill=COLORS["muted"], anchor="middle"))

    entry_y = price_y(float(signal_row["entry_price"]))
    parts.append(svg_text(signal_x + 10, entry_y - 10, "signal", size=11, weight="700", fill="#444"))
    parts.append(svg_line(signal_x + 2, entry_y - 4, signal_x + 2, entry_y + 28, stroke="#444", width=1.0))
    if example.family == "B":
        parts.append(svg_text(signal_x - 60, entry_y - 24, "pullback rebound", size=11, fill=COLORS["muted"]))

    caption = f"Historical example: {example.ticker} on {example.signal_date} ({example.pattern}, signal score {fmt_num(signal_row.get('signal_score'))})."
    parts.append(svg_text(MARGIN_LEFT, HEIGHT - 26, caption, size=12, fill=COLORS["muted"]))
    parts.append("</svg>")

    (OUT_DIR / example.output_name).write_text("\n".join(parts), encoding="utf-8")
    return caption


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICES_PATH, parse_dates=["Date"])
    signals = pd.read_csv(SIGNALS_PATH, parse_dates=["signal_date"])
    print("Generated charts:")
    for example in EXAMPLES:
        caption = build_chart(example, prices, signals)
        print(f"{example.family}|{example.output_name}|{caption}")


if __name__ == "__main__":
    main()