# Stock Triggers Documentation

This folder is for building daily swing-trading trigger signals from end-of-day prices.

Current status:

- Data pull scripts are ready.
- Trigger pattern logic (Pattern A and later patterns) can now be built on top of prices_eod.csv.

## Folder structure

- stock_triggers/scripts
  - update_prices_yf.py: Pulls OHLCV from Yahoo Chart API using requests.
  - update_prices_bhavcopy.py: Pulls OHLCV from NSE bhavcopy archives.
  - yfinance_probe.py: Utility/probe script for manual Yahoo endpoint checks.
- stock_triggers/data
  - prices_eod.csv: Canonical output used by trigger engine.
- stock_triggers/docs
  - data-source.md: This documentation.

## Canonical output file

Both updaters write/merge to:

- stock_triggers/data/prices_eod.csv

Schema:

- Date (YYYY-MM-DD)
- Ticker (e.g., RELIANCE.NS)
- Open
- High
- Low
- Close
- AdjClose
- Volume

Rows are deduplicated by Date + Ticker (latest value kept).

## Preferred source: Yahoo Chart API (direct requests)

Use script:

- stock_triggers/scripts/update_prices_yf.py

Why this script:

- Uses direct Yahoo chart endpoint via requests (not yfinance download internals).
- Supports custom User-Agent and request pacing to reduce throttling issues.

Run from repo root:

```bash
export SSL_CERT_FILE=/Users/Z0045SY/VisualStudioRepos/stock-operator/tgt-ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE

stockpy11/bin/python stock_triggers/scripts/update_prices_yf.py \
  --user-agent Brilliant \
  --tickers RELIANCE.NS TCS.NS INFY.NS \
  --days 365 \
  --pause-seconds 1.0
```

Options:

- --overwrite: replace prices_eod.csv instead of merging.
- --insecure: disable TLS verification (only for temporary debugging).
- --pause-seconds: delay between tickers to reduce Yahoo throttling.

## Alternative source: NSE bhavcopy

Use script:

- stock_triggers/scripts/update_prices_bhavcopy.py

Run from repo root:

```bash
stockpy11/bin/python stock_triggers/scripts/update_prices_bhavcopy.py \
  --tickers RELIANCE.NS TCS.NS INFY.NS \
  --start 2025-01-01 --end 2026-03-15
```

Or auto-read tickers from stock_selector/data/stocks.csv:

```bash
stockpy11/bin/python stock_triggers/scripts/update_prices_bhavcopy.py \
  --start 2025-01-01 --end 2026-03-15
```

Notes:

- Filters to SERIES=EQ.
- Tickers are matched by symbol and suffix (default .NS).

## Suggested daily workflow

1. Activate environment:

```bash
source stockpy11/bin/activate
```

2. Update prices (Yahoo direct requests preferred):

```bash
export SSL_CERT_FILE=/Users/Z0045SY/VisualStudioRepos/stock-operator/tgt-ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE

python stock_triggers/scripts/update_prices_yf.py \
  --user-agent Brilliant \
  --tickers RELIANCE.NS TCS.NS INFY.NS \
  --days 365
```

3. Use stock_triggers/data/prices_eod.csv as input to trigger pattern scripts.
