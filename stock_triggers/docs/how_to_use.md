# Stock Triggers – How To Use

This guide describes the daily workflow for updating prices and generating
Pattern A signals.

## 1. Activate environment and SSL bundle

From the repo root:

```bash
source stockpy11/bin/activate

export SSL_CERT_FILE=/Users/Z0045SY/VisualStudioRepos/stock-operator/tgt-ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
```

## 2. Update prices for your universe

The universe is defined once in stock_triggers/data/universe_tickers.txt (one
ticker per line). To refresh one year of history for all symbols and overwrite
prices_eod.csv:

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

## 3. Generate Pattern A triggers

To scan the latest available date in prices_eod.csv and produce Pattern A
breakout signals:

```bash
python stock_triggers/scripts/generate_triggers_pattern_a.py
```

Example with explicit date and parameters:

```bash
python stock_triggers/scripts/generate_triggers_pattern_a.py \
  --as-of-date 2026-03-13 \
  --breakout-days 40 \
  --volume-multiplier 1.5 \
  --stop-pct 7.0
```

This writes/overwrites:

- stock_triggers/data/signals_pattern_a.csv

## 4. Review signals

Open signals_pattern_a.csv and inspect:

- signal_date, ticker, pattern (e.g., A_breakout_40d)
- entry_price, entry_band_low, entry_band_high
- stop_price (7% below entry by default)

Use this list to shortlist trades, then confirm on charts before acting.
