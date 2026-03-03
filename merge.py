#!/usr/bin/env python3
"""
merge.py

Goal:
- Take your existing long-format S&P 500 price file (1996–2018)
- Add *only* the missing 2018–2023 data from the wide adj-close file
  (sp500_adjclose_2005_2023.csv)
- Use the daily membership file (sp500_membership_1990_2023_clean.csv)
  to set in_sp500 for the new rows
- Save a clean 1996–2023 long file with minimal survivorship bias.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG: update these names if your files are named differently
# ---------------------------------------------------------------------

# Candidates for your existing long file (1996–2018)
BASE_LONG_CANDIDATES = [
    "sp500_prices_1996_2018_long.csv",   # what your old script expected
    "sp500_prices_1996_2023_long.csv",   # what you actually seem to have
    "sp500_prices_long_1996_2018.csv",
]

# Wide adj-close file (2005–2023) that you sent the columns for
WIDE_2005_2023 = "sp500_adjclose_2005_2023.csv"

# Daily membership file with columns: date, tickers
MEMBERSHIP_DAILY = "sp500_membership_1990_2023_clean.csv"

# Output merged file
OUTPUT_MERGED = "sp500_prices_1996_2023_long_merged.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_existing_base_long(candidates):
    """
    Try a list of possible filenames for the existing 1996–2018 long file.
    Returns the first one that exists, or raises a clear error.
    """
    for fname in candidates:
        if os.path.exists(fname):
            print(f"[INFO] Using existing base long price file: {fname}")
            return fname

    msg = (
        "[ERROR] Could not find an existing 1996–2018 long price file.\n"
        "Looked for:\n  - "
        + "\n  - ".join(candidates)
        + "\n\nFix options:\n"
        "  1) Rename your existing long file to one of the above names, OR\n"
        "  2) Edit BASE_LONG_CANDIDATES in merge.py to match your actual file name."
    )
    raise FileNotFoundError(msg)


def load_existing_long(path):
    """
    Load the existing long-format price file.
    Expected columns at least: date, ticker, adj_close
    (Any extra columns are kept but ignored for the merge logic.)
    """
    print(f"[STEP] Loading existing long prices from: {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(
            f"[ERROR] Existing long file {path} must have 'date' and 'ticker' columns. "
            f"Columns found: {list(df.columns)}"
        )

    # If adj_close isn't present, try to infer it from 'close'
    if "adj_close" not in df.columns:
        if "adjclose" in df.columns:
            df.rename(columns={"adjclose": "adj_close"}, inplace=True)
        elif "close" in df.columns:
            df["adj_close"] = df["close"]
        else:
            raise ValueError(
                f"[ERROR] Existing long file {path} must have 'adj_close' or 'close' column. "
                f"Columns found: {list(df.columns)}"
            )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    print(f"[INFO] Existing long data date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[INFO] Existing long data tickers: {df['ticker'].nunique()} unique")

    return df


def load_wide_adjclose(path):
    """
    Load the wide adj-close 2005–2023 file:
    Columns: Date, A, AAPL, ABBV, ...
    Return melted long: date, ticker, adj_close
    """
    print(f"[STEP] Loading wide adj-close data from: {path}")
    df = pd.read_csv(path)

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]

    # The first column should be the dates (they told you it was 'Date')
    date_col = df.columns[0]
    if date_col.lower() != "date":
        print(f"[WARN] First column is '{date_col}', renaming to 'Date'.")
    df.rename(columns={date_col: "Date"}, inplace=True)

    df["date"] = pd.to_datetime(df["Date"])

    # Melt to long format
    value_cols = [c for c in df.columns if c not in ["Date", "date"]]
    long_df = df.melt(
        id_vars="date",
        value_vars=value_cols,
        var_name="ticker",
        value_name="adj_close"
    )

    long_df["ticker"] = long_df["ticker"].astype(str).str.upper().str.strip()
    long_df["adj_close"] = pd.to_numeric(long_df["adj_close"], errors="coerce")
    long_df = long_df.dropna(subset=["adj_close"])

    print(f"[INFO] Wide adj-close melted to long: {len(long_df):,} rows")
    print(f"[INFO] Date range: {long_df['date'].min().date()} → {long_df['date'].max().date()}")
    print(f"[INFO] Tickers: {long_df['ticker'].nunique()} unique")

    return long_df


def build_membership_long(path):
    """
    Build daily membership long-format from sp500_membership_1990_2023_clean.csv
    with columns: date, tickers (pipe-separated).
    Returns: date, ticker, in_sp500 (1 if member, else 0).
    """
    print(f"[STEP] Loading daily membership from: {path}")
    df = pd.read_csv(path)

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "tickers" not in df.columns:
        raise ValueError(
            f"[ERROR] Membership file {path} must have 'date' and 'tickers' columns. "
            f"Columns found: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].fillna("")

    # Split 'tickers' column by '|' and explode
    df_long = (
        df.assign(ticker=df["tickers"].str.split("|"))
          .explode("ticker")
    )

    df_long["ticker"] = df_long["ticker"].astype(str).str.upper().str.strip()
    df_long = df_long[df_long["ticker"] != ""]

    df_long["in_sp500"] = 1

    print(f"[INFO] Built membership long: {len(df_long):,} rows")
    print(f"[INFO] Date range: {df_long['date'].min().date()} → {df_long['date'].max().date()}")
    print(f"[INFO] Unique tickers in membership: {df_long['ticker'].nunique()}")

    return df_long[["date", "ticker", "in_sp500"]]


# ---------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------

def main():
    # 1. Find and load existing 1996–2018 long data
    base_path = find_existing_base_long(BASE_LONG_CANDIDATES)
    df_old = load_existing_long(base_path)

    # 2. Load 2005–2023 wide adj-close and melt to long
    if not os.path.exists(WIDE_2005_2023):
        raise FileNotFoundError(
            f"[ERROR] Wide adj-close file not found: {WIDE_2005_2023}\n"
            f"Make sure it's in the same folder as merge.py."
        )
    df_new_all = load_wide_adjclose(WIDE_2005_2023)

    # 3. Keep only NEW dates after the last date in df_old
    cutoff_date = df_old["date"].max()
    print(f"[STEP] Using cutoff date from existing data: {cutoff_date.date()}")

    df_new = df_new_all[df_new_all["date"] > cutoff_date].copy()
    print(f"[INFO] New rows after cutoff: {len(df_new):,}")

    if df_new.empty:
        print("[WARN] No new data found after cutoff date. "
              "You might already have 1996–2023 in your base file.")
        # Still write a copy so you at least get a known-good file
        df_old.sort_values(["ticker", "date"], inplace=True)
        df_old.to_csv(OUTPUT_MERGED, index=False)
        print(f"[DONE] Wrote merged file (unchanged) to: {OUTPUT_MERGED}")
        return

    # 4. Build daily membership long and set in_sp500 for new rows
    if not os.path.exists(MEMBERSHIP_DAILY):
        raise FileNotFoundError(
            f"[ERROR] Membership file not found: {MEMBERSHIP_DAILY}\n"
            "This is needed to avoid survivorship bias in 2018–2023."
        )

    mem_long = build_membership_long(MEMBERSHIP_DAILY)

    # left join: if a (date,ticker) is found in membership, in_sp500=1 else 0
    df_new = df_new.merge(
        mem_long,
        on=["date", "ticker"],
        how="left"
    )
    df_new["in_sp500"] = df_new["in_sp500"].fillna(0).astype(int)

    print(f"[INFO] New rows with membership merged: {len(df_new):,}")

    # 5. Harmonize columns with df_old
    # Ensure df_old has 'in_sp500' too; if missing, assume 1 (since it was built from membership)
    if "in_sp500" not in df_old.columns:
        df_old["in_sp500"] = 1

    # Keep a minimal set of columns that you actually need downstream
    keep_cols = ["date", "ticker", "adj_close", "in_sp500"]

    df_old_min = df_old[keep_cols].copy()
    df_new_min = df_new[keep_cols].copy()

    # 6. Concatenate and sort
    df_merged = pd.concat([df_old_min, df_new_min], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["date", "ticker"]).sort_values(
        ["ticker", "date"]
    )

    print(f"[INFO] Final merged rows: {len(df_merged):,}")
    print(f"[INFO] Final date range: {df_merged['date'].min().date()} → {df_merged['date'].max().date()}")
    print(f"[INFO] Final tickers: {df_merged['ticker'].nunique()} unique")

    # 7. Write to CSV
    df_merged.to_csv(OUTPUT_MERGED, index=False)
    print(f"[DONE] Wrote merged file to: {OUTPUT_MERGED}")


if __name__ == "__main__":
    main()