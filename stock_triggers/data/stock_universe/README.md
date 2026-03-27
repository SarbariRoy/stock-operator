# Stock Universe

This folder holds index constituent CSVs that feed the **Add more stocks** dropdown in the app.

Any `.csv` file placed here with a `Symbol`, `Ticker`, or `ticker` column is automatically picked up.

---

## Refreshing ind_nifty50list.csv

The file is the official Nifty 50 constituents list published by NSE India.

### Option 1 – Download from the NSE website

1. Go to **https://www.nseindia.com/market-data/live-equity-market**
2. Select **NIFTY 50** from the index dropdown.
3. Click the **Download (.csv)** button (top-right of the table).
4. Save the file here as `ind_nifty50list.csv`.

### Option 2 – Direct archive link

Download from the NSE archives:

```
https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv
```

> **Note:** NSE may block direct downloads without a browser session. If you get a 403 error, open the link in a browser first, or use a tool like `curl` with a referer header:
>
> ```bash
> curl -o ind_nifty50list.csv \
>   -H "Referer: https://www.nseindia.com" \
>   "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
> ```

### Other indices

You can add more index files the same way. Common ones:

| Index | Archive URL |
|-------|------------|
| Nifty Next 50 | `https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv` |
| Nifty 100 | `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv` |
| Nifty 200 | `https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv` |
| Nifty 500 | `https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv` |
| Nifty Midcap 150 | `https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv` |
| Nifty Smallcap 250 | `https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv` |

All these CSVs have a `Symbol` column, so the app picks them up automatically.
