# Stock Selector – Overview & Data

This project focuses on a single tool: a stock selector that ranks stocks using simple factor scores (momentum, profitability, valuation, volatility) and suggests how to allocate a fixed budget across the best ideas.

It works entirely from one CSV file of stock fundamentals and returns: stock_selector/data/stocks.csv.

## What it does

- Reads stocks.csv and drops illiquid/tiny names based on minimum price, market cap, and average volume.
- Builds factor scores per stock: momentum (6M/12M returns), profitability (ROE), valuation (P/E, optionally P/B), and risk (1Y volatility if available).
- Normalizes each factor to 0–1 and combines them into a composite_score.
- Ranks stocks by composite_score, picks the top N, and allocates a user‑specified budget across them in proportion to their scores.
- Prints a table with ticker, name, key metrics, composite_score, rupee allocation, and an approximate expected return.

## Where you can use it

- Personal screening of Indian equities (NSE) or any other market, as long as you can prepare a suitable CSV.
- Building a short list of stocks to research further, ranked by simple, transparent rules.
- Quickly re‑running screens after updating prices/fundamentals from a broker export or data provider.

## Input data: stocks.csv
| Column | Definition | Usefulness |
| --- | --- | --- |
| Ticker | Stock symbol (append .NS for NSE if using NSE) | <span style="color:green">High</span> |
| Name | Company name | <span style="color:orange">Moderate</span> |
| Sector | Sector / industry | <span style="color:orange">Moderate</span> |
| Price | Latest close price | <span style="color:green">High</span> |
| MarketCap | Market capitalization | <span style="color:green">High</span> |
| AvgVolume | Average daily trading volume | <span style="color:green">High</span> |
| PE | Trailing P/E (lower is cheaper) | <span style="color:green">High</span> |
| PB | Price-to-book ratio | <span style="color:orange">Moderate</span> |
| ROE | Return on equity (%) | <span style="color:green">High</span> |
| Return_6M | 6-month price return (%) | <span style="color:green">High</span> |
| Return_12M | 12-month price return (%) | <span style="color:green">High</span> |
| Volatility_1Y | Annualized daily volatility (%) over lookback window | <span style="color:green">High</span> |
| Source | Where the data came from (e.g., yfinance, broker export) | <span style="color:orange">Moderate</span> |

The script expects this file at stock_selector/data/stocks.csv by default, but you can point it to any CSV path via --file.

## How to run it

From the project root (the folder that contains stock_selector/), run:

python stock_selector/scripts/stock-selector.py \
	--budget 50000 \
	--top-n 10

Key options (all optional):

- --file: path to your CSV (default: stock_selector/data/stocks.csv).
- --budget: total budget in rupees to allocate across the selected stocks.
- --top-n: how many top stocks to include in the plan.
- --min-price, --min-mcap, --min-avg-volume: basic filters for liquidity/size.
- Column overrides like --ticker-col, --price-col, --mcap-col, --roe-col, --r6m-col, --r12m-col, --vol-col if your CSV uses different headers.

The output is a ranked table showing the suggested rupee allocation per stock and an approximate expected return based on the 12‑month return column.
