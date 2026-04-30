"""Quick test: verify all patterns detect signals and enhancers work."""
import sys, pandas as pd
sys.path.insert(0, ".")

prices = pd.read_csv("stock_triggers/data/st_lt_prices_eod.csv", parse_dates=["Date"])
latest = prices["Date"].max()
print(f"Latest date: {latest.date()}, Tickers: {prices['Ticker'].nunique()}")

from stock_triggers.ui.patterns.pattern_a import detect as a_det
from stock_triggers.ui.patterns.pattern_b import detect as b_det
from stock_triggers.ui.patterns.pattern_c_macd import detect as c_det
from stock_triggers.ui.patterns.pattern_d_rsi import detect as d_det
from stock_triggers.ui.patterns.pattern_e_boll import detect as e_det
from stock_triggers.ui.patterns.pattern_f_vwap import detect as f_det

# Run each pattern on the last 5 dates to accumulate some signals
dates = sorted(prices["Date"].unique())[-5:]
for name, fn, kw in [
    ("A breakout", a_det, dict(breakout_days=40, volume_multiplier=1.5, stop_pct=7.0)),
    ("B pullback", b_det, dict(volume_multiplier=1.5, stop_pct=7.0)),
    ("C MACD",     c_det, dict(volume_multiplier=1.0, stop_pct=7.0)),
    ("D RSI",      d_det, dict(volume_multiplier=1.0, stop_pct=7.0)),
    ("E Boll",     e_det, dict(volume_multiplier=1.0, stop_pct=7.0)),
    ("F VWAP",     f_det, dict(volume_multiplier=1.2, stop_pct=7.0)),
]:
    total = 0
    tickers_seen = set()
    for d in dates:
        df = fn(prices, as_of_date=pd.Timestamp(d), **kw)
        total += len(df)
        if not df.empty:
            tickers_seen.update(df["ticker"].tolist())
    sample = sorted(tickers_seen)[:5]
    print(f"  {name}: {total} signals across 5 days  sample={sample}")

# Enhancers
from stock_triggers.ui.enhancers.dragonfly_doji import check as doji_ck
from stock_triggers.ui.enhancers.hammer import check as hmr_ck
from stock_triggers.ui.enhancers.morning_star import check as ms_ck
from stock_triggers.ui.enhancers.bullish_engulfing import check as eng_ck

tickers = prices[prices["Date"] == latest]["Ticker"].unique()
doji_ct = sum(1 for t in tickers if doji_ck(prices, t))
hmr_ct = sum(1 for t in tickers if hmr_ck(prices, t))
ms_ct = sum(1 for t in tickers if ms_ck(prices, t))
eng_ct = sum(1 for t in tickers if eng_ck(prices, t))
print(f"\nEnhancers on {latest.date()}: doji={doji_ct}, hammer={hmr_ct}, morning_star={ms_ct}, engulfing={eng_ct}")
print("\nDONE")
