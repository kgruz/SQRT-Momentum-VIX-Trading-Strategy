#!/usr/bin/env python3
"""
data.py

Builds a survivorship-bias-free S&P 500 price panel (1996–2023)
using ONLY local CSVs:

1) sp500_membership_1990_2023_clean.csv
   - Columns: date, tickers
   - 'tickers' is a comma-separated string of tickers in the index on that date.

2) historical_stock_prices.csv.zip
   - Columns: ticker, open, close, adj_close, low, high, volume, date
   - Long format, many stocks, ~1996 onward.

3) sp500_adjclose_2005_2023.csv
   - WIDE format with columns:
       Date, A, AAPL, ABBV, ABNB, ...
   - Each ticker column contains adjusted close (adj_close).

Output:

- sp500_prices_1996_2023_long.csv
    columns: date, ticker, open, close, adj_close, low, high, volume, in_sp500

- sp500_prices_1996_2023_panel.csv
    index: date, columns: ticker, values: adj_close
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

MEMBERSHIP_CSV = "sp500_membership_1990_2023_clean.csv"
HISTORICAL_ZIP = "historical_stock_prices.csv"  # or .csv.zip as you have it
SP500_ADJ_2005_2023_CSV = "sp500_adjclose_2005_2023.csv"

OUT_LONG = "sp500_prices_1996_2023_long.csv"
OUT_PANEL = "sp500_prices_1996_2023_panel.csv"

START_DATE = "1996-01-01"
END_DATE = "2023-12-31"


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def check_file_exists(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path_str}")
    return path


# -------------------------------------------------------------------
# STEP 1 — MEMBERSHIP: wide ('tickers') → long ('date','ticker')
# -------------------------------------------------------------------

def load_membership(path_str: str) -> pd.DataFrame:
    """
    Load sp500_membership_1990_2023_clean.csv and convert from wide to long:
        Input columns:  date, tickers
            - 'tickers' is a comma-separated string of tickers active that date

        Output columns: date, ticker
            - One row per (date, ticker) membership
    """
    print("=== STEP 1: Load membership (wide → long) ===")
    check_file_exists(path_str)
    print(f"Loading membership from: {path_str}")

    df = pd.read_csv(path_str)

    if "date" not in df.columns or "tickers" not in df.columns:
        raise KeyError(
            "Expected columns ['date', 'tickers'] in membership CSV. "
            f"Found: {df.columns.tolist()}"
        )

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Drop rows with no tickers
    df = df.dropna(subset=["tickers"])

    # Split tickers on comma, strip spaces, uppercase
    df["tickers"] = df["tickers"].astype(str).str.split(",")

    df_long = df.explode("tickers").rename(columns={"tickers": "ticker"})

    # Clean ticker strings
    df_long["ticker"] = df_long["ticker"].astype(str).str.upper().str.strip()
    df_long = df_long[df_long["ticker"] != ""]  # remove empty

    # Drop duplicates just in case
    df_long = df_long.drop_duplicates(subset=["date", "ticker"]).reset_index(drop=True)

    # Filter to our desired date range
    df_long = df_long[
        (df_long["date"] >= START_DATE) & (df_long["date"] <= END_DATE)
    ].copy()

    print(f"Membership columns: {df_long.columns.tolist()}")
    print(
        f"Total unique S&P 500 tickers (ever in index {START_DATE}–{END_DATE}): "
        f"{df_long['ticker'].nunique()}"
    )
    print("Sample membership (long-format):")
    print(df_long.head(10))

    return df_long


# -------------------------------------------------------------------
# STEP 2 — HISTORICAL PRICES: historical_stock_prices.csv.zip (long)
# -------------------------------------------------------------------

def load_historical_prices(path_str: str) -> pd.DataFrame:
    """
    Load long-format historical price data from historical_stock_prices.csv.zip.

    Expected columns:
        ticker, open, close, adj_close, low, high, volume, date
    """
    print("\n=== STEP 2: Load historical price data (long) ===")
    check_file_exists(path_str)
    print(f"Loading historical prices from: {path_str}")

    df = pd.read_csv(path_str)

    expected_cols = {
        "ticker",
        "open",
        "close",
        "adj_close",
        "low",
        "high",
        "volume",
        "date",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(
            "historical_stock_prices.csv is missing expected columns: "
            f"{missing}. Found: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Filter date range
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Ensure numeric types
    numeric_cols = ["open", "close", "adj_close", "low", "high", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no adj_close or ticker
    df = df.dropna(subset=["adj_close", "ticker", "date"])

    print(f"Historical prices columns: {df.columns.tolist()}")
    print(f"Historical prices rows: {len(df):,}")
    print("Sample historical prices:")
    print(df.head(10))

    return df


# -------------------------------------------------------------------
# STEP 3 — SP500_ADJ_2005_2023: WIDE → LONG
# -------------------------------------------------------------------

def load_sp500_adj_prices(path_str: str) -> pd.DataFrame:
    """
    Load sp500_adjclose_2005_2023.csv and return a LONG-format dataframe with:
        columns: date, ticker, adj_close

    Your file is WIDE, with columns:
        Date, A, AAPL, ABBV, ABNB, ...

    We:
        - Rename 'Date' (or 'date') → 'date'
        - Treat every other column as a ticker's adj_close
        - Melt to long format.
    """
    print("\n=== STEP 3: Load S&P-specific adj_close data (2005–2023, wide → long) ===")
    check_file_exists(path_str)
    print(f"Loading S&P adj_close data from: {path_str}")

    df_raw = pd.read_csv(path_str)
    cols = df_raw.columns.tolist()
    print(f"S&P adj CSV columns (first 10): {cols[:10]}")

    # Find the date column, case-insensitive
    date_col = None
    for cand in ["date", "Date", "DATE"]:
        if cand in df_raw.columns:
            date_col = cand
            break

    if date_col is None:
        raise KeyError(
            "Expected a 'Date' or 'date' column in sp500_adjclose_2005_2023.csv. "
            f"Found: {df_raw.columns.tolist()}"
        )

    # Normalize column name to 'date'
    df_raw = df_raw.rename(columns={date_col: "date"})
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # All other columns are tickers (adj_close values)
    ticker_cols = [c for c in df_raw.columns if c != "date"]
    if not ticker_cols:
        raise KeyError(
            "Could not find any ticker columns in sp500_adjclose_2005_2023.csv "
            "besides the date column."
        )

    # Melt wide → long
    df_long = df_raw.melt(
        id_vars="date",
        value_vars=ticker_cols,
        var_name="ticker",
        value_name="adj_close"
    )

    df_long["ticker"] = df_long["ticker"].astype(str).str.upper().str.strip()

    # Filter date range (this file should be 2005–2023, but we'll still clamp)
    df_long = df_long[
        (df_long["date"] >= START_DATE) & (df_long["date"] <= END_DATE)
    ].copy()

    df_long["adj_close"] = pd.to_numeric(df_long["adj_close"], errors="coerce")
    df_long = df_long.dropna(subset=["adj_close"])

    print(f"S&P adj_close (2005–2023) rows: {len(df_long):,}")
    print("Sample S&P adj_close long-format:")
    print(df_long.head(10))

    return df_long


# -------------------------------------------------------------------
# STEP 4 — Combine price sources, prioritizing S&P-specific adj_close
# -------------------------------------------------------------------

def combine_price_sources(
    hist: pd.DataFrame,
    sp500_adj: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine:
        - hist: full historical price data (long, with open/close/adj_close/low/high/volume)
        - sp500_adj: S&P-specific adj_close for 2005–2023 (long: date,ticker,adj_close)

    Rules:
        - Keep all price fields from 'hist'
        - Where sp500_adj has a non-null adj_close for (date,ticker),
          overwrite hist.adj_close with this "better" S&P-specific value.

    Output: dataframe with
        columns: date, ticker, open, close, adj_close, low, high, volume
    """
    print("\n=== STEP 4: Combine historical prices with S&P-specific adj_close ===")

    # Ensure keys and types align
    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist["ticker"] = hist["ticker"].astype(str).str.upper().str.strip()

    sp500_adj = sp500_adj.copy()
    sp500_adj["date"] = pd.to_datetime(sp500_adj["date"])
    sp500_adj["ticker"] = sp500_adj["ticker"].astype(str).str.upper().str.strip()

    # Merge on (date, ticker) - left join on hist, overwrite adj_close if available
    sp500_adj = sp500_adj[["date", "ticker", "adj_close"]].rename(
        columns={"adj_close": "adj_close_sp500"}
    )

    merged = hist.merge(
        sp500_adj,
        on=["date", "ticker"],
        how="left",
        validate="m:1"
    )

    # Overwrite adj_close where we have an S&P-specific one
    merged["adj_close"] = np.where(
        merged["adj_close_sp500"].notna(),
        merged["adj_close_sp500"],
        merged["adj_close"]
    )

    merged = merged.drop(columns=["adj_close_sp500"])

    print(f"Combined price rows: {len(merged):,}")
    print("Sample combined prices:")
    print(merged.head(10))

    return merged


# -------------------------------------------------------------------
# STEP 5 — Join membership to prices → S&P-only panel, save outputs
# -------------------------------------------------------------------

def build_master():
    """
    Full pipeline:
        1) Load membership (long)
        2) Load historical prices
        3) Load S&P-specific adj_close (2005–2023, wide → long)
        4) Combine price sources
        5) Inner-join membership with prices → S&P-only data
        6) Save long and panel formats
    """
    # 1) Membership
    membership = load_membership(MEMBERSHIP_CSV)

    # 2) Historical prices (long)
    hist = load_historical_prices(HISTORICAL_ZIP)

    # 3) S&P adj_close patch (2005–2023, wide → long)
    sp500_adj = load_sp500_adj_prices(SP500_ADJ_2005_2023_CSV)

    # 4) Combine / overwrite adj_close where S&P-specific data exists
    combined_prices = combine_price_sources(hist, sp500_adj)

    # 5) Restrict to tickers that ever appear in membership (just to shrink universe)
    sp500_tickers = membership["ticker"].unique()
    combined_prices = combined_prices[
        combined_prices["ticker"].isin(sp500_tickers)
    ].copy()

    # 6) Join membership with prices: only keep dates where the stock is in the S&P 500
    print("\n=== STEP 5: Join membership with prices (S&P-only) ===")

    # Add in_sp500 flag = 1 for membership rows
    membership["in_sp500"] = 1

    master = combined_prices.merge(
        membership[["date", "ticker", "in_sp500"]],
        on=["date", "ticker"],
        how="inner",          # keeps only dates when the ticker was in the index
        validate="m:1"
    )

    master = master.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Final S&P-only rows (1996–2023): {len(master):,}")
    print(f"Unique tickers in final master: {master['ticker'].nunique()}")
    print("Sample of final master (long):")
    print(master.head(15))

    # 7) Save LONG-format
    print(f"\nSaving LONG-format data to: {OUT_LONG}")
    master.to_csv(OUT_LONG, index=False)

    # 8) Save PANEL-format (date × ticker → adj_close)
    print(f"Building PANEL-format adj_close matrix: {OUT_PANEL}")
    panel = master.pivot(index="date", columns="ticker", values="adj_close")
    panel = panel.sort_index()
    panel.to_csv(OUT_PANEL)

    print("\nDone.")
    print(f"  Long-format file : {OUT_LONG}")
    print(f"  Panel-format file: {OUT_PANEL}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    build_master()