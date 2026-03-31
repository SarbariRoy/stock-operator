# Stock Universe Helper Files

This folder is for index-constituent CSV files that the app can pick up automatically.

In plain language, these files are just convenience lists.

They help the UI offer quick “add more stocks” choices without you typing every ticker one by one.

## What kind of files work here

Any CSV placed in this folder is usable if it has one of these columns:

- Symbol
- Ticker
- ticker

## Current example

- ind_nifty50list.csv

That is the Nifty 50 constituent list.

## How the app uses these files

```mermaid
flowchart LR
		A[index CSV in stock_universe folder] --> B[app detects file]
		B --> C[user picks an index list]
		C --> D[tickers can be added into the tracked universe flow]
```

## Refreshing ind_nifty50list.csv

### Option 1: download from NSE in the browser

1. Open the NSE equity market page.
2. Choose NIFTY 50.
3. Download the CSV.
4. Save it here as ind_nifty50list.csv.

### Option 2: use the archive link

```text
https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv
```

If NSE blocks the direct request, use a browser session first or try curl with a referer header.

Example:

```bash
curl -o ind_nifty50list.csv \
	-H "Referer: https://www.nseindia.com" \
	"https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
```

## Other useful index files

- Nifty Next 50
- Nifty 100
- Nifty 200
- Nifty 500
- Nifty Midcap 150
- Nifty Smallcap 250

If those CSVs use a Symbol column, the app can usually pick them up without extra code changes.
