"""Download the daily prices used in Part 4 (VaR and Expected Shortfall).

Run from the repository root:

    python scripts/download_data.py

Source: Yahoo Finance via the yfinance package, adjusted close prices
(auto_adjust=True: prices adjusted for splits and dividends), in EUR.

The data is saved to data/prices.csv, which is NOT committed: Yahoo Finance
data should not be redistributed, and anyone can regenerate it with this
script. If the download fails, the script stops with an error; it never
falls back to synthetic data.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Five CAC 40 stocks from different sectors, all quoted in EUR on Euronext Paris.
TICKERS = {
    "MC.PA": "LVMH (luxury)",
    "TTE.PA": "TotalEnergies (energy)",
    "SAN.PA": "Sanofi (healthcare)",
    "BNP.PA": "BNP Paribas (banking)",
    "AIR.PA": "Airbus (aerospace)",
}
START = "2012-01-01"

# Absolute daily log returns above this level are flagged for manual review.
EXTREME_RETURN = 0.25


def download() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(list(TICKERS), start=START, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no data (network problem, rate limit or ticker change)"
        )
    prices = raw["Close"]
    missing = [t for t in TICKERS if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise RuntimeError(f"no data returned for: {missing}")
    return prices[list(TICKERS)]


def main() -> None:
    try:
        prices = download()
    except Exception as exc:  # report any failure clearly and stop
        print(f"Download FAILED: {exc}", file=sys.stderr)
        print(
            "No file was written. Do not replace the data with synthetic prices.", file=sys.stderr
        )
        sys.exit(1)

    n_raw = len(prices)
    # Align dates: keep only days on which all five stocks have a price.
    prices = prices.dropna(how="any").sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    # Sanity checks.
    if (prices <= 0).any().any():
        raise ValueError("non-positive prices found, check the data")
    log_returns = np.log(prices).diff().dropna()
    extreme = log_returns[(log_returns.abs() > EXTREME_RETURN).any(axis=1)]

    DATA.mkdir(exist_ok=True)
    prices.to_csv(DATA / "prices.csv")

    print(
        f"Downloaded on {date.today().isoformat()} from Yahoo Finance (yfinance, auto_adjust=True)"
    )
    for ticker, name in TICKERS.items():
        print(f"  {ticker:7s} {name}")
    print(
        f"Rows downloaded: {n_raw}, rows with all prices: {len(prices)} "
        f"(dropped {n_raw - len(prices)})"
    )
    print(
        f"Start: {prices.index[0].date()}, end: {prices.index[-1].date()}, "
        f"observations: {len(prices)} prices, {len(log_returns)} daily returns"
    )
    if extreme.empty:
        print(f"No absolute daily log return above {EXTREME_RETURN:.0%}.")
    else:
        print(f"Flagged for manual review (|log return| > {EXTREME_RETURN:.0%}), kept in the data:")
        print(extreme.round(4).to_string())
    print(f"Saved {(DATA / 'prices.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
