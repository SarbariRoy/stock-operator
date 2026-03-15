# stock-operator

Tools for researching and operating a personal stock portfolio. The repo
currently has two main pieces:

- A **stock selector** that ranks stocks based on fundamentals and momentum.
- A **stock triggers** engine that generates swing-trade entry signals from
  end-of-day price data.

## Layout

- stock_selector/
  - scripts/stock-selector.py – score and rank stocks from a CSV universe.
  - data/ – input data for the selector (e.g., stocks.csv).
  - docs/ – documentation for how the selector works and how to run it.

- stock_triggers/
  - scripts/ – data updaters and pattern scripts (Pattern A).
  - data/ – price history (prices_eod.csv), universe file, and signals CSVs.
  - docs/ – how the trigger engine works, data sources, and pattern details.

- requirements.txt – pinned Python dependencies for the project.

## Python environment

Create or reuse a virtualenv (stockpy11 is the one used in examples):

```bash
python3 -m venv stockpy11
source stockpy11/bin/activate
pip install -r requirements.txt
```

On this machine, a custom CA bundle is used for HTTPS:

```bash
export SSL_CERT_FILE=/path/to/tgt-ca-bundle.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE
```

Adjust the path to your local CA bundle or remove these lines if you don\'t
need them.

## Using the stock selector

High-level flow (see stock_selector/docs/data-documentation.md for details):

1. Prepare/update stock_selector/data/stocks.csv with your universe and
   fundamental/momentum fields.
2. Run the selector script to score and rank:

   ```bash
   python stock_selector/scripts/stock-selector.py
   ```

3. Review the ranked output and choose which names to keep in your universe.

## Using the stock triggers engine

High-level flow (see stock_triggers/README.md and docs):

1. Define your trading universe once in
   stock_triggers/data/universe_tickers.txt (one ticker per line).
2. Update end-of-day OHLCV prices for the universe:

   ```bash
   python stock_triggers/scripts/update_prices_yf.py \
     --user-agent Brilliant \
     --days 365 \
     --pause-seconds 0.8 \
     --overwrite \
     --universe-file stock_triggers/data/universe_tickers.txt
   ```

3. Generate Pattern A breakout triggers from the latest data:

   ```bash
   python stock_triggers/scripts/generate_triggers_pattern_a.py
   ```

4. Open stock_triggers/data/signals_pattern_a.csv, review the signals, and
   then consult charts before deciding any trades.

## Automation (GitHub Actions)

The repo includes a scheduled workflow:

- .github/workflows/daily-triggers.yml
- .github/workflows/notify-existing-signals.yml

What it does:

1. Refreshes prices from the universe file.
2. Generates Pattern A triggers.
3. Commits updated CSV outputs back to the repo.
4. Optionally sends Telegram update if secrets are configured.

The lightweight notify-only workflow:

1. Does not refresh prices.
2. Does not generate triggers.
3. Sends Telegram using existing signals file (`--send-only`).

How to use it:

1. Push this repo to GitHub.
2. Enable Actions for the repository.
3. (Optional, for phone alerts) add GitHub repository secrets:
  - TELEGRAM_BOT_TOKEN
  - TELEGRAM_CHAT_ID
4. Run once manually via Actions -> Daily Triggers Pipeline -> Run workflow.

By default it is scheduled Mon-Fri at 13:30 UTC (about 19:00 IST).

## Next steps / ideas

- Connect the selector and triggers more tightly (only run patterns on
  top-ranked names).
- Add more patterns (B, C, ...) and document them in
  stock_triggers/docs/patterns.md.
- Add simple backtests on top of historical signals to understand behaviour
  before using with real capital.
