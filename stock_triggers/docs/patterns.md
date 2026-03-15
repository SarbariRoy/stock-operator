# Patterns Overview

This document describes the trading patterns implemented (or planned) in the
stock_triggers workspace: how they are defined, how they are detected, and
what the key parameters mean.

Right now, only **Pattern A** is implemented. Future patterns (B, C, ...) can
follow a similar structure and will be added here as they are built.

---

## Pattern A – Trend Breakout With Volume

**Goal:** Identify stocks that are already in an uptrend and are breaking out
to a new high, with strong volume confirming the move.

### 1. How Pattern A is defined

Pattern A is defined using daily OHLCV data and moving averages:

- Uptrend filter:
  - 50-day simple moving average of Close (SMA50) is greater than the
    200-day simple moving average of Close (SMA200).
- Price position:
  - Close > SMA50 and Close > SMA200.
- Breakout condition:
  - Close is greater than the highest Close of the previous N trading days
    (not including today). By default N = 40.
- Volume confirmation:
  - Today\'s Volume is at least `volume_multiplier` times the average Volume of
    the last 20 trading days. By default `volume_multiplier` = 1.5.

If **all** of these conditions are true for a stock on the as-of date, Pattern
A fires a buy signal.

Formally, for each stock on a given date T:

- `SMA50(T) = mean(Close over last 50 days up to T)`
- `SMA200(T) = mean(Close over last 200 days up to T)`
- `PrevNHighClose(T) = max(Close over previous N days before T)`
- `VolAvg20(T) = mean(Volume over last 20 days up to T)`

Pattern A requires:

- `SMA50(T) > SMA200(T)`
- `Close(T) > SMA50(T)`
- `Close(T) > SMA200(T)`
- `Close(T) > PrevNHighClose(T)`
- `Volume(T) >= volume_multiplier * VolAvg20(T)`

### 2. How Pattern A is detected in code

Implementation: [stock_triggers/scripts/generate_triggers_pattern_a.py](stock_triggers/scripts/generate_triggers_pattern_a.py)

High-level detection steps for a given as-of date:

1. Load prices from prices_eod.csv (Date, Ticker, Open, High, Low, Close,
   AdjClose, Volume).
2. For each Ticker:
   - Sort rows by Date.
   - Compute rolling indicators:
     - `SMA50` = 50-day rolling mean of Close.
     - `SMA200` = 200-day rolling mean of Close.
     - `VolAvg20` = 20-day rolling mean of Volume.
     - `PrevNHighClose` = rolling max of Close over `breakout_days`, shifted by 1
       day so that today\'s breakout is measured against **prior** closes.
3. Take the row for the as-of date. If any of SMA50, SMA200, VolAvg20,
   PrevNHighClose are missing (not enough history), skip this stock.
4. Evaluate the Pattern A conditions listed above.
5. If Pattern A passes:
   - Set `entry_price` = today\'s Close.
   - Set `entry_band_low` = `entry_price` (same as Close for now).
   - Set `entry_band_high` = `entry_price * 1.02` (2% band above Close).
   - Set `stop_price` = `entry_price * (1 - stop_pct/100)` (default 7% below).
6. Write one row into signals_pattern_a.csv with key fields:
   - `signal_date` – as-of date.
   - `ticker` – stock symbol.
   - `pattern` – e.g., `A_breakout_40d`.
   - `close`, `sma50`, `sma200`, `prev_high_close`.
   - `volume`, `vol_avg20`.
   - `entry_price`, `entry_band_low`, `entry_band_high`.
   - `stop_pct`, `stop_price`.

### 3. Why this is a potential buy point

The intuition behind Pattern A:

- **Trend filter (SMA50 > SMA200):**
  - Focuses only on stocks in a confirmed longer-term uptrend.
  - Avoids trying to buy breakouts in downtrends or sideways markets.
- **Price above both moving averages:**
  - Confirms that the stock is trading above key support zones.
  - Suggests recent strength rather than a one-day spike from a low base.
- **Breakout to a new N-day high:**
  - Indicates the stock is doing something it hasn\'t done in a while (e.g.,
    making a 40-day high).
  - Helps catch momentum as the stock potentially enters a stronger phase.
- **Volume spike vs 20-day average:**
  - Confirms that the breakout is backed by higher-than-normal participation.
  - Reduces the chance that the move is just a low-volume head-fake.
- **Stop loss below entry:**
  - Defines risk upfront (e.g., 7% below entry).
  - If price fails quickly after the breakout, you exit before a bigger loss.

Put together, Pattern A looks for a strong stock, in an uptrend, making a fresh
high, with unusual volume behind the move, and defines where you are wrong.

### 4. Parameters and what they control

Pattern A is controlled by a few key parameters, exposed as command-line
arguments in generate_triggers_pattern_a.py:

- `--breakout-days` (default: 40)
  - Meaning:
    - The number of **previous trading days** used to define the breakout
      high. Close today must be greater than the highest Close over these
      days.
  - Effect of increasing:
    - Harder to trigger (you need to beat a longer history of prices).
    - Signals are rarer but often more significant (e.g., 60-day highs).
  - Effect of decreasing:
    - Easier to trigger (e.g., 20-day highs fire more often).
    - More signals, but some may be more "noisy" or shorter-term in nature.

- `--volume-multiplier` (default: 1.5)
  - Meaning:
    - The multiple of the 20-day average Volume needed today to count as a
      volume spike.
    - Condition: `Volume_today >= volume_multiplier * VolAvg20`.
  - Effect of increasing:
    - Demands stronger volume confirmation.
    - Fewer signals, but those that pass have more extreme volume.
  - Effect of decreasing:
    - Allows breakouts on more modest volume.
    - More signals, but some may not have strong institutional
      participation behind them.

- `--stop-pct` (default: 7.0)
  - Meaning:
    - The percentage distance between entry price and initial stop loss.
    - Stop is placed at `entry_price * (1 - stop_pct/100)`.
  - Effect of increasing (e.g., 10%):
    - Wider stop ⇒ more room for volatility/breathing space.
    - Higher risk per trade if the stop is hit.
  - Effect of decreasing (e.g., 5%):
    - Tighter stop ⇒ you exit faster if the breakout fails.
    - Lower per-trade risk, but higher chance of being stopped out by normal
      noise.

- `--as-of-date` (default: latest date in prices_eod.csv)
  - Meaning:
    - The date on which to evaluate the pattern.
  - Use cases:
    - Omit it for live/end-of-day usage (default = most recent date in
      prices_eod.csv).
    - Set it explicitly to backtest specific dates or to re-run signals for a
      past day.

### 5. Changing the parameters – practical guidance

For a **conservative, high-quality** signal set:

- Use a larger breakout window (e.g., `--breakout-days 40` or 60).
- Keep or even increase `--volume-multiplier` (e.g., 1.5–2.0).
- Use a moderate stop (e.g., `--stop-pct 7.0`).

This will give you fewer, but often stronger, breakouts.

For a **more active, experimental** setup (more signals, more noise):

- Use a shorter breakout window (e.g., `--breakout-days 20`).
- Reduce the volume threshold slightly (e.g., `--volume-multiplier 1.0–1.2`).
- You might also tighten stops (e.g., `--stop-pct 5.0`) to control risk.

You can run generate_triggers_pattern_a.py multiple times with different
parameter sets and compare how many signals you get and whether they look
sensible on charts.

---

As new patterns (B, C, etc.) are implemented, they can be documented here with
similar sections: definition, detection logic, intuition, parameters, and how
changing those parameters affects behaviour.
