# Stock Triggers – What This Workspace Is Doing

This folder is the "trigger engine" that sits on top of your raw price data.

- It collects daily OHLCV data for a fixed universe of stocks into
  stock_triggers/data/prices_eod.csv.
- It runs pattern-based rules (starting with Pattern A breakout) on that
  price history to generate swing-trade entry signals.
- Its outputs are simple CSVs you can review by eye or plug into other tools.

## Components

- stock_triggers/scripts/update_prices_yf.py
  - Pulls daily OHLCV from the Yahoo Chart API (via requests).
  - Reads your stock universe from stock_triggers/data/universe_tickers.txt or
    from the --tickers argument.
- stock_triggers/scripts/generate_triggers_pattern_a.py
  - Reads prices_eod.csv and applies Pattern A (trend + breakout + volume).
  - Writes signals to stock_triggers/data/signals_pattern_a.csv.
- stock_triggers/scripts/update_prices_bhavcopy.py
  - Alternative source using NSE bhavcopy archives.
- stock_triggers/scripts/yfinance_probe.py
  - Utility probe for manual data endpoint checks.
- stock_triggers/docs/data-source.md
  - Detailed data-source and workflow documentation.
