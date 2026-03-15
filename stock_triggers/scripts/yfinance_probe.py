"""Simple yfinance probe script.

Runs a set of common yfinance endpoints for a single ticker and prints samples.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import yfinance as yf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run basic yfinance queries for a ticker")
    parser.add_argument("--ticker", type=str, default="MSFT", help="Ticker symbol (default: MSFT)")
    parser.add_argument("--period", type=str, default="1mo", help="History period (default: 1mo)")
    args = parser.parse_args()

    dat = yf.Ticker(args.ticker)

    print(f"Ticker: {args.ticker}")

    print("\ninfo:")
    try:
        pprint(dat.info)
    except Exception as exc:
        print(f"info failed: {exc}")

    print("\ncalendar:")
    try:
        print(dat.calendar)
    except Exception as exc:
        print(f"calendar failed: {exc}")

    print("\nanalyst_price_targets:")
    try:
        print(dat.analyst_price_targets)
    except Exception as exc:
        print(f"analyst_price_targets failed: {exc}")

    print("\nquarterly_income_stmt:")
    try:
        print(dat.quarterly_income_stmt)
    except Exception as exc:
        print(f"quarterly_income_stmt failed: {exc}")

    print(f"\nhistory(period='{args.period}'):")
    try:
        print(dat.history(period=args.period))
    except Exception as exc:
        print(f"history failed: {exc}")

    print("\noption_chain(first expiry).calls:")
    try:
        options = dat.options
        if not options:
            print("No option expiries available for this ticker.")
        else:
            print(dat.option_chain(options[0]).calls)
    except Exception as exc:
        print(f"option_chain failed: {exc}")


if __name__ == "__main__":
    main()
