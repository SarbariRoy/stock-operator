# Stock Triggers

Daily swing-trading trigger workspace built on end-of-day OHLCV data.

## What this workspace is doing

This folder is the "trigger engine" that sits on top of your raw price data.

- It collects daily OHLCV data for a fixed universe of stocks into
  [stock_triggers/data/prices_eod.csv](stock_triggers/data/prices_eod.csv).
- It runs pattern-based rules (starting with Pattern A breakout) on that
  price history to generate swing-trade entry signals.
- Its outputs are simple CSVs you can review by eye or plug into other tools.

Key pieces:

- [stock_triggers/scripts/update_prices_yf.py](stock_triggers/scripts/update_prices_yf.py)
  - Pulls daily OHLCV from the Yahoo Chart API (via requests).
  - Reads your stock universe from
    [stock_triggers/data/universe_tickers.txt](stock_triggers/data/universe_tickers.txt)
    or from `--tickers` on the command line.
- [stock_triggers/scripts/generate_triggers_pattern_a.py](stock_triggers/scripts/generate_triggers_pattern_a.py)
  - Reads `prices_eod.csv` and applies Pattern A (trend + breakout + volume).
  - Writes signals to
    [stock_triggers/data/signals_pattern_a.csv](stock_triggers/data/signals_pattern_a.csv).
- [stock_triggers/scripts/update_prices_bhavcopy.py](stock_triggers/scripts/update_prices_bhavcopy.py)
  - Alternative source using NSE bhavcopy archives.
- [stock_triggers/scripts/yfinance_probe.py](stock_triggers/scripts/yfinance_probe.py)
  - Utility probe for manual data endpoint checks.
- [stock_triggers/docs/data-source.md](stock_triggers/docs/data-source.md)
  - Detailed data-source and workflow documentation.
 - stock_triggers/ui/
   - Streamlit-based UI scripts for visualizing triggers
     (see stock_triggers/ui/README.md).

## How to use it (daily flow)

From the repo root:

1. **Activate environment and SSL bundle**

   ```bash
   source stockpy11/bin/activate

   export SSL_CERT_FILE=/Users/Z0045SY/VisualStudioRepos/stock-operator/tgt-ca-bundle.crt
   export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
   export CURL_CA_BUNDLE=$SSL_CERT_FILE
   ```

2. **Update prices for your universe (30 stocks)**

   The universe is defined once in
   [stock_triggers/data/universe_tickers.txt](stock_triggers/data/universe_tickers.txt)
   (one ticker per line). To refresh one year of history for all of them and
   overwrite the existing prices file:

   ```bash
   python stock_triggers/scripts/update_prices_yf.py \
     --user-agent Brilliant \
     --days 365 \
     --pause-seconds 0.8 \
     --overwrite \
     --universe-file stock_triggers/data/universe_tickers.txt
   ```

   This writes/updates:

   - stock_triggers/data/prices_eod.csv

3. **Generate Pattern A triggers**

   To scan the latest available date in `prices_eod.csv` and produce Pattern A
   breakout signals:

   ```bash
   python stock_triggers/scripts/generate_triggers_pattern_a.py
   ```

   Or, to run for a specific date with custom parameters (example):

   ```bash
   python stock_triggers/scripts/generate_triggers_pattern_a.py \
     --as-of-date 2026-03-13 \
     --breakout-days 40 \
     --volume-multiplier 1.5 \
     --stop-pct 7.0
   ```

   This writes/overwrites:

   - stock_triggers/data/signals_pattern_a.csv

4. **Review the signals**

   Open `signals_pattern_a.csv` and look at:

   - `signal_date`, `ticker`, `pattern` (e.g., `A_breakout_40d`)
   - `entry_price`, `entry_band_low`, `entry_band_high`
   - `stop_price` (7% below entry by default)

   You can then manually check charts and decide which trades (if any) to take
   the next day.

## Output schema

prices_eod.csv columns:

- Date
- Ticker
- Open
- High
- Low
- Close
- AdjClose
- Volume

Rows are deduplicated by Date + Ticker.

## Alternative source (bhavcopy)

```bash
python stock_triggers/scripts/update_prices_bhavcopy.py \
  --tickers RELIANCE.NS TCS.NS INFY.NS \
  --start 2025-01-01 --end 2026-03-15
```

## Next step

Use stock_triggers/data/prices_eod.csv as input for Pattern A trigger generation.
