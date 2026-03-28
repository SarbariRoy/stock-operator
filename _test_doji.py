"""Quick test to find dragonfly doji shaped candles in the price data."""
import pandas as pd

prices = pd.read_csv("stock_triggers/data/prices_eod.csv")
prices["Date"] = pd.to_datetime(prices["Date"])

count = 0
for ticker, g in prices.groupby("Ticker"):
    g = g.sort_values("Date")
    g["SMA50"] = g["Close"].rolling(50).mean()
    g["SMA200"] = g["Close"].rolling(200).mean()
    for _, r in g.tail(60).iterrows():
        o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
        rng = h - l
        if rng <= 0:
            continue
        body = abs(c - o) / rng
        lower = (min(o, c) - l) / rng
        upper = (h - max(o, c)) / rng
        if body <= 0.3 and lower >= 0.6 and upper <= 0.15:
            sma50 = r.get("SMA50")
            sma200 = r.get("SMA200")
            trend = "UP" if pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200 else "DOWN"
            near50 = ""
            if pd.notna(sma50) and sma50 > 0:
                dist = abs(c - sma50) / sma50 * 100
                near50 = f"dist_sma50={dist:.1f}%"
            print(f"{str(r['Date'])[:10]}  {ticker:20s}  body={body:.2f} lower={lower:.2f} upper={upper:.2f}  trend={trend}  {near50}")
            count += 1

print(f"\nTotal T-shaped candles (shape only, last 60 days): {count}")
